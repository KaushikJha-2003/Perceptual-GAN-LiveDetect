import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.config import cfg

class PerceptualGAN_Dataset(Dataset):
    def __init__(self, root, annotations_file, transform=None):
        self.root = root
        self.transform = transform
        self.annotations = self._load_annotations(annotations_file)
        self.class_mapping = self._get_class_mapping()
        self.image_names = list(self.annotations.keys()) # Store image names as a list

    def _load_annotations(self, annotations_file):
        annotations = {}
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                img_name = parts[0]
                if img_name not in annotations:
                    annotations[img_name] = []
                # Format: [xmin, ymin, xmax, ymax, class_id]
                box = [int(p) for p in parts[1:5]]
                label = int(parts[5])

                # Validate and clip bounding box coordinates
                img_path = os.path.join(self.root, img_name)
                try:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size

                    xmin, ymin, xmax, ymax = box

                    # Clip coordinates to be within image boundaries
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_width - 1, xmax)
                    ymax = min(img_height - 1, ymax)

                    # Ensure xmin < xmax and ymin < ymax
                    if xmin < xmax and ymin < ymax:
                         annotations[img_name].append({'bbox': [xmin, ymin, xmax, ymax], 'label': label})
                    else:
                        print(f"Skipping invalid bounding box for {img_name}: {box}")

                except FileNotFoundError:
                    print(f"Warning: Image file not found for annotation: {img_name}")
                    # Still include the image name in annotations even if file is not found
                    # This will lead to an error later, but prevents the dataset from being empty initially.
                    if img_name not in annotations:
                         annotations[img_name] = []
                except Exception as e:
                    print(f"Error processing annotation for {img_name}: {e}")
                    # Still include the image name in annotations even if there's an error
                    if img_name not in annotations:
                         annotations[img_name] = []


        return annotations

    def _get_class_mapping(self):
        class_ids = set()
        for img_name in self.annotations:
            for annotation in self.annotations[img_name]:
                class_ids.add(annotation['label'])

        sorted_ids = sorted(list(class_ids))
        class_mapping = {label: i + 1 for i, label in enumerate(sorted_ids)}
        class_mapping[0] = 0 # Background class
        return class_mapping

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root, img_name)
        image = Image.open(img_path).convert("RGB")
        image_width, image_height = image.size

        targets = []
        boxes = []
        labels = []
        if img_name in self.annotations:
            for ann in self.annotations[img_name]:
                bbox = ann['bbox']
                label = ann['label']
                boxes.append(bbox)
                labels.append(label)


        if self.transform:
            try:
                augmented = self.transform(image=np.array(image), bboxes=boxes, class_labels=labels)
                image = augmented['image']
                boxes = augmented['bboxes']
                labels = augmented['class_labels']
            except ValueError as e:
                print(f"Error augmenting image {img_name} with boxes {boxes}: {e}")
                raise e

        # Handle cases where no valid boxes remain after augmentation
        if len(boxes) == 0:
             boxes = torch.zeros((0, 4), dtype=torch.float32)
             labels = torch.zeros((0,), dtype=torch.int64)


        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)


        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['image_width'] = torch.tensor([image_width])
        target['image_height'] = torch.tensor([image_height])


        return image, target

    def __len__(self):
        return len(self.image_names)

# Augmentations for training
train_transform = A.Compose([
    A.Resize(height=480, width=640),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Augmentations for testing
test_transform = A.Compose([
    A.Resize(height=480, width=640),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def collate_fn(batch):
    return tuple(zip(*batch))
