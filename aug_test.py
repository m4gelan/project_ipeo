import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import os
import random
import numpy as np
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

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

def transform_train_with_labels(image, labels):
    """
    Apply geometric and image augmentations to the input image and labels.

    Args:
        image (PIL.Image): The input image.
        labels (torch.Tensor): Bounding boxes in YOLO format [class, x_center, y_center, width, height].

    Returns:
        augmented_images (list of PIL.Image): List of augmented images.
        augmented_labels (list of torch.Tensor): List of corresponding transformed labels.
    """
    augmented_images = []
    augmented_labels = []
    
    # Geometric transformations applied to both image and labels
    geometric_transforms = [
        ("vertical_flip", lambda img, lbl: (F.vflip(img), flip_y(lbl))),
        ("horizontal_flip", lambda img, lbl: (F.hflip(img), flip_x(lbl))),
        ("rotate_90", lambda img, lbl: (F.rotate(img, 90), rotate_90(lbl))),
        ("rotate_180", lambda img, lbl: (F.rotate(img, 180), rotate_180(lbl))),
        ("rotate_270", lambda img, lbl: (F.rotate(img, 270), rotate_270(lbl)))
    ]

    # Randomly apply 2-3 geometric transformations
    selected_transforms = random.sample(geometric_transforms, random.randint(2, 3))
    for _, transform in selected_transforms:
        aug_img, aug_lbl = transform(image, labels)
        augmented_images.append(aug_img)
        augmented_labels.append(aug_lbl)

    # Additional image-only transformations
    if random.random() < 0.5:
        img_transform = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        augmented_images = [img_transform(img) for img in augmented_images]
    
    return augmented_images, augmented_labels

# Helper functions for label transformations
def flip_x(labels):
    """Flip labels horizontally."""
    if labels.numel() == 0:
        return labels
    labels[:, 1] = 1 - labels[:, 1]
    return labels

def flip_y(labels):
    """Flip labels vertically."""
    if labels.numel() == 0:
        return labels
    labels[:, 2] = 1 - labels[:, 2]
    return labels

def rotate_90(labels):
    """Rotate labels 90 degrees."""
    if labels.numel() == 0:
        return labels
    x, y = labels[:, 1], labels[:, 2]
    labels[:, 1] = y
    labels[:, 2] = 1 - x
    return labels

def rotate_180(labels):
    """Rotate labels 180 degrees."""
    if labels.numel() == 0:
        return labels
    labels[:, 1] = 1 - labels[:, 1]
    labels[:, 2] = 1 - labels[:, 2]
    return labels

def rotate_270(labels):
    """Rotate labels 270 degrees."""
    if labels.numel() == 0:
        return labels
    x, y = labels[:, 1], labels[:, 2]
    labels[:, 1] = 1 - y
    labels[:, 2] = x
    return labels

def save_augmented_data(augmented_images, augmented_labels, original_image_name, output_image_folder, output_label_folder, augmentation_names):
    """
    Save augmented images and their YOLOv8 labels.

    Args:
        augmented_images (list of PIL.Image): List of augmented images.
        augmented_labels (list of torch.Tensor): List of augmented labels (YOLOv8 format).
        original_image_name (str): Original image filename (e.g., "image_1.jpg").
        output_image_folder (str): Directory to save augmented images.
        output_label_folder (str): Directory to save augmented labels.
        augmentation_names (list of str): Names of applied augmentations (e.g., "horizontal_flip").
    """
    # Create output directories if they don't exist
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    for idx, (image, labels, aug_name) in enumerate(zip(augmented_images, augmented_labels, augmentation_names)):
        # Construct file names
        base_name = os.path.splitext(original_image_name)[0]
        augmented_image_name = f"{base_name}_{aug_name}_{idx + 1}.jpg"
        augmented_label_name = f"{base_name}_{aug_name}_{idx + 1}.txt"

        # Save augmented image
        image_path = os.path.join(output_image_folder, augmented_image_name)
        image.save(image_path, format="JPEG")

        # Save augmented labels
        label_path = os.path.join(output_label_folder, augmented_label_name)
        with open(label_path, "w") as f:
            for label in labels:
                f.write(" ".join(map(str, label.tolist())) + "\n")

    print(f"Saved augmented data for {original_image_name}.")

class RockDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, label_folder, mean, std, transform=None, augment=False):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
        self.label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.txt')])
        self.transform = transform
        self.normalize = T.Normalize(mean=mean, std=std)
        self.augment = augment  # Enable augmentation

        assert len(self.image_files) == len(self.label_files)

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
        
        if content:
            labels = [list(map(float, line.split())) for line in content.split("\n")]
            labels_tensor = torch.tensor(labels)
        else:
            labels_tensor = torch.empty((0, 5))

        # Apply augmentation if enabled
        if self.augment:
            augmented_images, augmented_labels = transform_train_with_labels(image, labels_tensor)
            return augmented_images, augmented_labels

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