import torch
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import rasterio as rio
import os
import random
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def calculate_mean_std(image_folder):
    """
    Calculate the mean and standard deviation of the dataset
    """
    pixel_sum = torch.zeros(3)
    pixel_squared_sum = torch.zeros(3)
    num_pixels = 0

    # Iterate through all files in the folder
    for file_name in os.listdir(image_folder):
        if file_name.endswith(".jpg"):  # .jpg files
            image_path = os.path.join(image_folder, file_name)
            
            # Open image
            image = Image.open(image_path).convert("RGB")
            
            # Convert to tensor
            image_tensor = T.ToTensor()(image)  # Shape: [C, H, W]
            
            # Update pixel sums
            pixel_sum += image_tensor.sum(dim=(1, 2))
            pixel_squared_sum += (image_tensor ** 2).sum(dim=(1, 2))
            num_pixels += image_tensor.size(1) * image_tensor.size(2)  # H * W

    # Calculate mean and std
    mean = pixel_sum / num_pixels
    std = (pixel_squared_sum / num_pixels - mean ** 2).sqrt()

    return mean, std

class RockDetectionDataset(Dataset):
    def __init__(self, image_folder, label_folder, mean, std, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
        self.label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.txt')])
        self.transform = transform
        self.normalize = T.Normalize(mean=mean, std=std)

        assert len(self.image_files) == len(self.label_files), "Number of images and labels must match!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        image_tensor = T.ToTensor()(image)
        image_tensor = self.normalize(image_tensor)

        # Load labels
        label_path = os.path.join(self.label_folder, self.label_files[idx])
        with open(label_path, "r") as f:
            content = f.read().strip()
        
        if content:  # Check if the label file is not empty
            labels = [list(map(float, line.split())) for line in content.split("\n")]
            labels_tensor = torch.tensor(labels)
        else:
            labels_tensor = torch.empty((0, 5))  # Empty tensor for no rocks

        return image_tensor, labels_tensor
    
def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized labels, including empty labels.
    Args:
        batch: List of (image, label) tuples from the dataset.
    Returns:
        Tuple of batched images and a list of label tensors.
    """
    images = torch.stack([item[0] for item in batch])  # Stack images into a batch
    labels = [item[1] for item in batch]  # Keep labels as a list of tensors
    return images, labels