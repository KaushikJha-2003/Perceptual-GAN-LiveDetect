import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw, ImageFont, Image

def save_checkpoint(model, path, epoch):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch']
    return 0

def draw_boxes(image_tensor, boxes, labels, class_names):
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    # Denormalize image for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)

    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image_pil)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        class_name = class_names.get(label.item(), "Unknown")

        color = 'red'
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        draw.text((xmin, ymin - 15), f"{class_name}", fill=color, font=font)

    return image_pil

def visualize_results(image, targets, detections, class_names, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    gt_image = draw_boxes(image, targets[0]['boxes'], targets[0]['labels'], class_names)
    ax1.imshow(gt_image)
    ax1.set_title('Ground Truth')
    ax1.axis('off')

    pred_boxes = detections[0]['boxes'].detach().cpu()
    pred_labels = detections[0]['labels'].detach().cpu()
    pred_scores = detections[0]['scores'].detach().cpu()

    confidence_threshold = 0.5
    valid_indices = pred_scores > confidence_threshold
    filtered_boxes = pred_boxes[valid_indices]
    filtered_labels = pred_labels[valid_indices]

    pred_image = draw_boxes(image, filtered_boxes, filtered_labels, class_names)
    ax2.imshow(pred_image)
    ax2.set_title('Predictions')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Visualization saved to {output_path}")
