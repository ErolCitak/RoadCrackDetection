import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import transforms as T


class DINOv2FasterRCNN(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super(DINOv2FasterRCNN, self).__init__()
        
        # Load DINOv2 backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
        # Remove the classification head from the backbone
        # This is crucial - we only want to keep the feature extractor part
        self.backbone.head = nn.Identity()
        
        # If using ViT, we need to ensure the backbone outputs feature maps
        # DINOv2 ViT models output feature maps with shape [B, C, H, W]
        # Assuming dinov2_vitb14 outputs features with 768 channels
        out_channels = 768
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # RPN needs to know the number of output channels
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        # Define the region proposal network
        rpn_head = torchvision.models.detection.rpn.RPNHead(
            out_channels, anchor_generator.num_anchors_per_location()[0]
        )
        
        # Define the ROI pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Define the box head
        box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
            out_channels * 7 * 7, 1024
        )
        
        # Define the box predictor
        box_predictor = FastRCNNPredictor(1024, num_classes)
        
        # Combine all components into the Faster R-CNN model
        self.model = FasterRCNN(
            backbone=self.backbone,
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            box_roi_pool=roi_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            min_size=800, max_size=1333  # Default image size range
        )
    
    def forward(self, images, targets=None):
        return self.model(images, targets)

# Example dataset class for object detection
class ObjectDetectionDataset(Dataset):
    def __init__(self, image_paths, annotations, transforms=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Load annotations
        # Format: [x1, y1, x2, y2, class_id]
        boxes = self.annotations[idx]["boxes"]
        labels = self.annotations[idx]["labels"]
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # Suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        return image, target

# Get transforms for training and validation
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Helper function to collate batches
def collate_fn(batch):
    return tuple(zip(*batch))

# Training function
def train_model(model, data_loader, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimize
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            running_loss += losses.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(data_loader):.4f}')
    
    return model

# Example usage
def main():
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    num_classes = 91  # COCO has 90 classes + background
    model = DINOv2FasterRCNN(num_classes=num_classes, freeze_backbone=True)
    
    # Example: create your dataset
    # You need to replace these with your actual data
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
    annotations = [
        {
            "boxes": [[100, 100, 200, 200], [300, 300, 400, 400]],
            "labels": [1, 2]
        },
        {
            "boxes": [[150, 150, 250, 250]],
            "labels": [3]
        }
    ]
    
    # Create dataset
    dataset = ObjectDetectionDataset(
        image_paths=image_paths,
        annotations=annotations,
        transforms=get_transform(train=True)
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Train model
    model = train_model(model, data_loader, optimizer, device, num_epochs=10)
    
    # Save the model
    torch.save(model.state_dict(), 'dinov2_faster_rcnn.pth')

if __name__ == "__main__":
    main()
