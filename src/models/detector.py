import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from configs.config import cfg

class Detector(nn.Module):
    def __init__(self, num_classes):
        super(Detector, self).__init__()

        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # Replace the box predictor with a new one that has the updated number of classes
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)
