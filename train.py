import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.ops import roi_align
import os
import random
import numpy as np
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Import custom modules
from src.data.dataset import PerceptualGAN_Dataset, train_transform, test_transform, collate_fn
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.models.detector import Detector
from src.utils.utils import save_checkpoint, load_checkpoint, visualize_results
from configs.config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(detector, generator, discriminator, data_loader, optimizers, epoch):
    detector.train()
    generator.train()
    discriminator.train()

    optim_d, optim_g, optim_det = optimizers

    total_loss_d = 0
    total_loss_g = 0
    total_loss_det = 0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(cfg.DEVICE) for img in images]
        targets = [{k: v.to(cfg.DEVICE) for k, v in t.items()} for t in targets]

        # --- Train Discriminator ---
        optim_d.zero_grad()

        large_object_boxes = []
        for t in targets:
            areas = (t['boxes'][:, 2] - t['boxes'][:, 0]) * (t['boxes'][:, 3] - t['boxes'][:, 1])
            large_indices = areas > 2000
            if large_indices.any():
                # Prepend image index (assuming batch size 1 for now for simplicity with roi_align on single feature map)
                # For batch size > 1, need to handle boxes from different images separately or use batched roi_align
                img_index = t['image_id'].item()
                boxes_with_index = torch.cat([torch.full((large_indices.sum(), 1), img_index, device=cfg.DEVICE), t['boxes'][large_indices]], dim=1)
                large_object_boxes.append(boxes_with_index)


        small_object_boxes = []
        for t in targets:
            areas = (t['boxes'][:, 2] - t['boxes'][:, 0]) * (t['boxes'][:, 3] - t['boxes'][:, 1])
            small_indices = areas < 1000
            if small_indices.any():
                 # Prepend image index (assuming batch size 1 for now for simplicity with roi_align on single feature map)
                # For batch size > 1, need to handle boxes from different images separately or use batched roi_align
                img_index = t['image_id'].item()
                boxes_with_index = torch.cat([torch.full((small_indices.sum(), 1), img_index, device=cfg.DEVICE), t['boxes'][small_indices]], dim=1)
                small_object_boxes.append(boxes_with_index)


        if not large_object_boxes or not small_object_boxes:
            continue

        # Concatenate boxes across the batch for roi_align
        all_large_boxes = torch.cat(large_object_boxes, dim=0)
        all_small_boxes = torch.cat(small_object_boxes, dim=0)


        real_features = detector.model.backbone(torch.stack(images)) # Pass the whole batch to the backbone
        real_features = real_features['0']
        real_features_rois = roi_align([real_features], all_large_boxes, (7, 7), 1/16)


        small_features = detector.model.backbone(torch.stack(images)) # Pass the whole batch to the backbone
        small_features = small_features['0']
        small_features_rois = roi_align([small_features], all_small_boxes, (7, 7), 1/16)


        generated_features = generator(small_features_rois)

        real_scores = discriminator(real_features_rois)
        fake_scores = discriminator(generated_features.detach())

        loss_d_real = -torch.mean(real_scores)
        loss_d_fake = torch.mean(fake_scores)
        loss_d = loss_d_real + loss_d_fake

        loss_d.backward()
        optim_d.step()

        # --- Train Generator ---
        optim_g.zero_grad()

        generated_features = generator(small_features_rois)
        fake_scores = discriminator(generated_features)
        loss_g = -torch.mean(fake_scores)

        loss_g.backward()
        optim_g.step()

        # --- Train Detector ---
        optim_det.zero_grad()

        losses = detector(images, targets)
        loss_det = sum(loss for loss in losses.values())

        loss_det.backward()
        optim_det.step()

        total_loss_d += loss_d.item()
        total_loss_g += loss_g.item()
        total_loss_det += loss_det.item()

    avg_loss_d = total_loss_d / len(data_loader)
    avg_loss_g = total_loss_g / len(data_loader)
    avg_loss_det = total_loss_det / len(data_loader)

    print(f"Epoch {epoch}/{cfg.NUM_EPOCHS} | D Loss: {avg_loss_d:.4f}, G Loss: {avg_loss_g:.4f}, Det Loss: {avg_loss_det:.4f}")

def main():
    set_seed(cfg.SEED)

    full_dataset = PerceptualGAN_Dataset(cfg.DATASET_PATH, cfg.ANNOTATION_FILE, transform=train_transform)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS
    )

    class_names = {v: k for k, v in full_dataset.dataset.class_mapping.items()}

    generator = Generator(in_channels=256, out_channels=256).to(cfg.DEVICE)
    discriminator = Discriminator(in_channels=256).to(cfg.DEVICE)
    detector = Detector(num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)

    optim_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.LEARNING_RATE_D)
    optim_g = torch.optim.Adam(generator.parameters(), lr=cfg.LEARNING_RATE_G)
    optim_det = torch.optim.Adam(detector.parameters(), lr=cfg.LEARNING_RATE_DETECTOR)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_one_epoch(detector, generator, discriminator, train_loader, (optim_d, optim_g, optim_det), epoch)

        if epoch % cfg.SAVE_EVERY_N_EPOCHS == 0 or epoch == cfg.NUM_EPOCHS:
            save_checkpoint(detector, f"{cfg.CHECKPOINT_DIR}/detector_epoch_{epoch}.pth", epoch)
            save_checkpoint(generator, f"{cfg.CHECKPOINT_DIR}/generator_epoch_{epoch}.pth", epoch)

    detector.eval()
    with torch.no_grad():
        test_images, test_targets = next(iter(test_loader))
        test_images = [img.to(cfg.DEVICE) for img in test_images]
        test_targets = [{k: v.to(cfg.DEVICE) for k, v in t.items()} for t in test_targets]

        detections = detector(test_images)

        visualize_results(
            test_images[0],
            test_targets,
            detections,
            class_names,
            os.path.join(cfg.RESULT_DIR, 'test_results.png')
        )

if __name__ == "__main__":
    main()
