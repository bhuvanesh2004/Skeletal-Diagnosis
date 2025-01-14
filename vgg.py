import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection.ssd import SSD
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Define the classes for X-ray diagnosis
CLASSES = [
    'background',  # class 0
    'elbow positive',
    'fingers positive', 
    'forearm fracture',
    'humerus fracture',
    'humerus',
    'shoulder fracture',
    'wrist positive'
]

def create_model(pretrained=True):
    """Create an SSD model with VGG16 backbone."""
    try:
        model = torchvision.models.detection.ssd300_vgg16(
            weights='DEFAULT' if pretrained else None,
            num_classes=len(CLASSES)  # Include background class
        )
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

def get_vgg_model():
    """Get VGG model with appropriate weights."""
    # First try to create the model
    model = create_model(pretrained=True)
    if model is None:
        print("Failed to create VGG model, trying fallback model...")
        try:
            # Try creating a basic pretrained model as fallback
            model = torchvision.models.detection.ssd300_vgg16(weights='DEFAULT')
            print(" Using general-purpose pretrained model - not suitable for medical diagnosis")
        except Exception as e:
            print(f"Failed to create fallback model: {e}")
            return None
    
    # Set model to evaluation mode
    model.eval()
    
    # Try to load custom weights if available
    weights_path = os.path.join('weights', 'model_vgg.pt')
    if os.path.exists(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            print("Loaded custom VGG weights for X-ray analysis")
        except Exception as e:
            print(" Error loading custom weights. Using pretrained weights not suitable for medical diagnosis")
            print(f"Error details: {e}")
    else:
        print(" No custom weights found. Using pretrained weights not suitable for medical diagnosis")
    
    return model

def plot_detection_results(img, prediction, threshold=0.5):
    """Plot detection results with proper class labels and confidence scores."""
    img = img.cpu().detach().permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.axis('off')
    
    class_name = None
    detection_info = None
    
    if prediction and "scores" in prediction and len(prediction["scores"]) > 0:
        # Get indices of detections above threshold
        keep_idx = prediction["scores"] > threshold
        boxes = prediction["boxes"][keep_idx]
        scores = prediction["scores"][keep_idx]
        labels = prediction["labels"][keep_idx]
        
        if len(scores) > 0:
            # Get the highest confidence detection
            max_score_idx = torch.argmax(scores)
            box = boxes[max_score_idx].detach().cpu().numpy()
            label_idx = int(labels[max_score_idx])
            score = float(scores[max_score_idx])
            
            # Ensure label_idx is within valid range
            if 0 <= label_idx < len(CLASSES):
                class_name = CLASSES[label_idx]
                
                # Plot the bounding box
                rect = patches.Rectangle(
                    (box[0], box[1]), (box[2] - box[0]), (box[3] - box[1]),
                    linewidth=2, edgecolor='orange', facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add warning about model limitations
                warning_text = " Demo Only - Not for Medical Use"
                ax.text(box[0], box[1] - 25, warning_text,
                       fontsize=10, color='red', fontweight='bold')
                
                ax.text(box[0], box[1] - 10, f"{class_name} ({score:.2f})",
                       fontsize=12, color='orange', fontweight='bold')
                
                # Store detection info
                detection_info = {
                    "class": class_name,
                    "confidence": f"{score:.2f}",
                    "box": box.tolist(),
                    "warning": "This is a demonstration only. Not suitable for medical diagnosis."
                }
            else:
                print(f"Warning: Invalid label index {label_idx}")
                class_name = "unknown"
    
    return fig, ax, class_name, detection_info

def figure_to_array(fig):
    """Convert matplotlib figure to numpy array."""
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)