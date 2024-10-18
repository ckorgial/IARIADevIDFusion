# -*- coding: utf-8 -*-
"""GRADCAM_FUSION.ipynb

Automatically generated by Colab.
"""

!pip install torch

from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define the ResNet18 model and modify the final layer to match num_classes=25
class ResNet18Model(nn.Module):
    def __init__(self, num_classes=25):
        super(ResNet18Model, self).__init__()
        self.model = models.resnet18(weights=None)
        # Replace the final fully connected layer with a new one (matching num_classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load the model and the checkpoint as ".ckpt"
checkpoint_path = ''
model = ResNet18Model()

# Load the checkpoint safely
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Preprocess the image
img_path = ''
image = Image.open(img_path).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

# Initialize gradients variable
gradients = None

# Define a hook to save gradients
def save_gradient(grad):
    global gradients
    gradients = grad

# Get the last convolutional layer
def get_last_conv_layer(model):
    # Look for the last convolutional layer in the model
    conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    return conv_layers[-1]

# Forward hook to capture the output of the last conv layer
class FeatureExtractorHook:
    def __init__(self, model):
        self.gradients = None
        self.features = None
        self.hook = None
        self.model = model

    def hook_layers(self):
        last_conv_layer = get_last_conv_layer(self.model)
        self.hook = last_conv_layer.register_forward_hook(self.save_feature_and_gradient)

    def save_feature_and_gradient(self, module, input, output):
        self.features = output  # Save the output of the conv layer
        output.register_hook(self.save_gradient)  # Register hook on the output to save gradients

    def save_gradient(self, grad):
        self.gradients = grad

    def remove_hook(self):
        if self.hook:
            self.hook.remove()

# Instantiate the feature extractor and hook the last conv layer
extractor = FeatureExtractorHook(model)
extractor.hook_layers()

# Forward pass to get predictions and save features
output = model(input_tensor)
pred = torch.argmax(output, dim=1)

# Backward pass to calculate gradients
model.zero_grad()
output[:, pred].backward()

# Extract the gradients and feature maps
gradients = extractor.gradients[0].detach().numpy()
feature_maps = extractor.features[0].detach().numpy()

# Perform global average pooling on the gradients
weights = np.mean(gradients, axis=(1, 2))

# Compute the Grad-CAM heatmap
gradcam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
for i, w in enumerate(weights):
    gradcam += w * feature_maps[i]

# Apply ReLU to the heatmap and normalize it
gradcam = np.maximum(gradcam, 0)
gradcam = gradcam - np.min(gradcam)
gradcam = gradcam / np.max(gradcam)

# Resize the heatmap to match the input image size
gradcam = cv2.resize(gradcam, (image.size[0], image.size[1]))

# Visualize the original image and the Grad-CAM heatmap
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Grad-CAM')
plt.imshow(image)
plt.imshow(gradcam, alpha=0.5, cmap='jet')
plt.axis('off')

plt.show()

# Clean up the hook to prevent memory leaks
extractor.remove_hook()