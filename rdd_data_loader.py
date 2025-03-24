import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import utils

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.v2 as T_v2
from torchvision.transforms.v2 import functional as F

from torchvision.transforms import v2
from torchvision import tv_tensors

def collate_fn(batch):
    return tuple(zip(*batch))

class RoadDamageDataset(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms
        self.class_2_idx = {'D00':0, 'D10':1, 'D20':2, 'D40':3, 'D43':4, 'D44':5, 'D50':6 }
        self.idx_2_class = {v: k for k, v in self.class_2_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        annot_path = self.data.iloc[idx]['path']
        image_path = annot_path.replace("\\annotations\\", "\\images\\").replace("\\xmls", "").replace(".xml", ".jpg").replace(".XML", ".jpg")

        img = Image.open(image_path).convert("RGB")

        # Get image dimensions for BoundingBoxes
        img_width, img_height = img.size  # Get dimensions from PIL Image
        
        # Read bounding boxes
        boxes = []
        labels = []

        annot_data = utils.parse_annotation(annot_path)
        for annot_object in annot_data["objects"]:
            name = annot_object['name']
            bbox = annot_object['bndbox']
            x1, x2, y1, y2 = bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax']


            if name in self.class_2_idx.keys():
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_2_idx[name])
            else:
                continue

        # Create BoundingBoxes with correct canvas size (height, width)
        boxes = tv_tensors.BoundingBoxes(
                boxes,
                format="XYXY", 
                canvas_size=(img_height, img_width)  # Pass (H, W) not the array shape
            )
        

        if self.transforms is not None:
            img, boxes = self.transforms(img, boxes)

        ## Some image labels are broken like;
        ##  AssertionError: All bounding boxes should have positive height and width. Found invalid box [198.0, 474.0, 198.0, 475.0] for target at index 9.
        for box in boxes:
            x1, y1, x2, y2 = box

            if (x2 <= x1):
                box[2] += 10
            if (y2 <= y1):
                box[3] += 10
                
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return img, target
    

if __name__ == '__main__':

    # Create a proper transform for object detection that transforms both images and targets
    def get_transform():

        # Define transforms
        my_transform = v2.Compose([
            v2.Resize(1024),
            v2.CenterCrop(896),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])

        
        return my_transform

    dino_transform = get_transform()
    
    dataset = RoadDamageDataset(csv_file='train_paths.csv', transforms=dino_transform)
    #dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    

    # Dataloader'dan bir batch çek ve test et
    for images, targets in dataloader:
        print(f"Batch Size: {len(images)}")
        print(f"Image Shape: {images[0].shape}")
        print(f"Target Example: {targets[0]}")
        print('-'*30)
        print(targets)

        break  # Sadece bir batch kontrol et