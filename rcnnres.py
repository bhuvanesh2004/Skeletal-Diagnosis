import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSD
import torch.nn as nn
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

classes = [
    'background',  # class 0
    'elbow positive',
    'fingers positive', 
    'forearm fracture',
    'humerus fracture',
    'humerus',
    'shoulder fracture',
    'wrist positive'
]

def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=len(classes))
    
    weights_path = os.path.join("weights", "Resnet.pt")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    else:
        print(f"Warning: Model weights not found at {weights_path}. Using pretrained weights.")
    
    return model

def get_vgg_model():
    model = torchvision.models.detection.ssd300_vgg16(weights='DEFAULT', num_classes=len(classes))
    weights_path = os.path.join("weights", "model_vgg.pt")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    else:
        print(f"Warning: Model weights not found at {weights_path}. Using pretrained weights.")
    
    return model

def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)):
        idx_list = []
        for idx, score in enumerate(preds[id]['scores']):
            if score > threshold:
                idx_list.append(idx)
        
        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]
    
    return preds

def plot_image_from_output(img, annotation):
    img = img.cpu().detach().permute(1, 2, 0).numpy()    
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.axis('off')
    
    class_name = None
    
    if annotation and "scores" in annotation and len(annotation["scores"]) > 0:
        max_score_idx = torch.argmax(annotation["scores"])
        
        # Extract the coordinates of the bounding box with the highest score
        xmin, ymin, xmax, ymax = annotation["boxes"][max_score_idx].detach().cpu().numpy()
        label_idx = int(annotation["labels"][max_score_idx])
        
        # Ensure label_idx is within valid range
        if 0 <= label_idx < len(classes):
            class_name = classes[label_idx]
            
            # Plot the bounding box with the highest score
            rect = patches.Rectangle(
                (xmin, ymin), (xmax - xmin), (ymax - ymin),
                linewidth=2, edgecolor='orange', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(xmin, ymin - 10, f"{class_name} ({annotation['scores'][max_score_idx]:.2f})",
                   fontsize=12, color='orange', fontweight='bold')
        else:
            print(f"Warning: Invalid label index {label_idx}")
            class_name = "unknown"

    return fig, ax, class_name

def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)