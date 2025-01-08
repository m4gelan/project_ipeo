import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
import os
import random
import numpy as np
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import json
import shutil
import re

# Dataset organisation functions:

def load_dataset_from_json(json_file_path):
    """
    Load the dataset from a JSON file.

    Parameters:
        json_file_path (str): Path to the JSON file.

    Returns:
        dict: The loaded JSON data.
        list: The dataset extracted from the JSON file.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    # print('General information about the data:', data.get('info', 'No info available'))
    return data, data.get('dataset', [])

def split_train_from_json(dataset, img_folder, base_dir_name):
    """
    Organize a dataset into train and test directories based on metadata.

    Parameters:
        dataset (list of dict): A list where each item contains metadata for a file.
                               Each dict should have 'file_name' and 'split' keys.
        img_folder (str): The folder containing the original images.
        base_dir (str): The base directory for the dataset structure.
    """
    # Define directories
    train_images = os.path.join(base_dir_name, 'train_images')
    test = os.path.join(base_dir_name, 'test')

    # Create directories if they don't exist
    if not os.path.exists(base_dir_name):
        os.makedirs(train_images)
        os.makedirs(test)

    # Iterate through all samples
    for sample in dataset:
        relative_file_name = sample.get('file_name')
        split = sample.get('split')  # Assuming the "split" key indicates train/test

        if not relative_file_name or not split:
            print(f"Missing data in sample: {sample}. Skipping.")
            continue

        # Construct full source path
        src_path = os.path.join(img_folder, relative_file_name)

        # Define destination directory
        if split == 'train':
            dest_dir = train_images
        elif split == 'test':
            dest_dir = test
        else:
            print(f"Unknown split '{split}' for file '{relative_file_name}'. Skipping.")
            continue

        dest_path = os.path.join(dest_dir, os.path.basename(relative_file_name))

        # Copy file to the appropriate directory
        try:
            shutil.copy(src_path, dest_path)
            # print(f"Copied '{relative_file_name}' to '{dest_dir}'")
        except Exception as e:
            print(f"Error copying '{relative_file_name}': {e}")

    print("Dataset split completed.")
    return train_images, test

def save_train_annotations(dataset, base_dir_name):
    """
    Save annotations for training images to individual .txt files.

    Parameters:
        dataset (list of dict): A list where each item contains metadata for a file.
                               Each dict should have 'file_name', 'split', and optionally 'rocks_annotations' keys.
        base_dir_name (str): The base directory where the train_labels folder will be created.
    """
    train_labels = os.path.join(base_dir_name, 'train_labels')

    # Create the train_labels directory if it doesn't exist
    os.makedirs(train_labels, exist_ok=True)

    # Process images with split == 'train'
    for sample in dataset:
        if sample.get('split') == 'train':
            # Extract relevant details
            file_name = sample.get('file_name')
            annotations = sample.get('rocks_annotations', [])

            # Create a .txt file for this image
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            txt_file_path = os.path.join(train_labels, f"{base_name}.txt")

            # Write annotations to the .txt file
            with open(txt_file_path, 'w') as txt_file:
                for annotation in annotations:
                    txt_file.write(f"{annotation}\n")

            # print(f"Created annotation file: {txt_file_path}")

    print("All train annotations have been saved to the 'train_labels' folder.")
    return train_labels

def create_validation_set_images(train_images_folder, base_dir_name):
    """
    Create a validation set by moving 10% of training images to a validation folder.

    Parameters:
        train_images_folder (str): Path to the training images folder.
        base_dir_name (str): Base directory name where validation images will be stored.
    """
    val_images = os.path.join(base_dir_name, 'val_images')
    os.makedirs(val_images, exist_ok=True)

    # List all files in the source folder
    files = [file for file in os.listdir(train_images_folder) if os.path.isfile(os.path.join(train_images_folder, file))]

    # Calculate 10% of the total files
    num_files_to_move = max(1, int(len(files) * 0.1))  # Ensure at least one file is moved

    # Randomly select 10% of the files
    files_to_move = random.sample(files, num_files_to_move)

    # Move the selected files
    for file in files_to_move:
        src_path = os.path.join(train_images_folder, file)
        dest_path = os.path.join(val_images, file)
        shutil.move(src_path, dest_path)
        # print(f"Moved '{file}' to '{val_images}'")

    print(f"Moved {len(files_to_move)} files to '{val_images}'.")
    return val_images

def create_validation_set_labels(train_labels_folder, base_dir_name):
    """
    Move corresponding label files for validation images to a validation labels folder.

    Parameters:
        train_labels_folder (str): Path to the training labels folder.
        base_dir_name (str): Base directory name where validation labels will be stored.
    """
    val_labels = os.path.join(base_dir_name, 'val_labels')
    val_images_folder = os.path.join(base_dir_name, 'val_images')
    os.makedirs(val_labels, exist_ok=True)

    # List all image files in val_images folder (excluding extensions)
    val_image_files = {os.path.splitext(file)[0] for file in os.listdir(val_images_folder) if os.path.isfile(os.path.join(val_images_folder, file))}

    # Move matching label files from train_labels to val_labels
    for label_file in os.listdir(train_labels_folder):
        # Get the base name (without extension) of the label file
        base_name = os.path.splitext(label_file)[0]

        if base_name in val_image_files:
            src_path = os.path.join(train_labels_folder, label_file)
            dest_path = os.path.join(val_labels, label_file)
            shutil.move(src_path, dest_path)
            # print(f"Moved '{label_file}' to '{val_labels}'")

    print("Matching label files moved to 'val_labels' folder.")
    return val_labels

def convert_tif_to_jpg(folder_path):
    """
    Convert all .tif files in a folder to .jpg format and remove the original .tif files.

    Parameters:
        folder_path (str): Path to the folder containing .tif images.
    """
    for file in os.listdir(folder_path):
        if file.endswith('.tif'):
            # Full path to the .tif file
            tif_path = os.path.join(folder_path, file)

            # Open the .tif file
            try:
                with Image.open(tif_path) as img:
                    # Define the output path with the same name but .jpg extension
                    jpg_path = os.path.join(folder_path, file.replace('.tif', '.jpg'))

                    # Convert and save as JPG
                    img.convert('RGB').save(jpg_path, 'JPEG')

                    # Remove the original .tif file
                    os.remove(tif_path)
                    # print(f"Converted and replaced: {file} -> {jpg_path}")

            except Exception as e:
                print(f"Error processing {file}: {e}")

    print("All .tif files have been converted to .jpg and replaced.")

def convert_labels_to_yolo_format(label_input_folder, base_dir_name, bbox_width, bbox_height, type):
    """
    Convert labels to YOLOv8 format.

    Parameters:
        label_input_folder (str): Path to the folder containing the original label files.
        base_dir_name (str): Base directory name where yolo labels will be stored.
        bbox_width (float): Normalized width of the bounding box.
        bbox_height (float): Normalized height of the bounding box.
        type (str): 'train' or 'val' fot the name of the folder.
    """
    label_output_folder = os.path.join(base_dir_name, f'yolo_{type}_labels')
    os.makedirs(label_output_folder, exist_ok=True)

    for label_file in os.listdir(label_input_folder):
        if label_file.endswith('.txt'):  # Process only text files
            input_path = os.path.join(label_input_folder, label_file)
            output_path = os.path.join(label_output_folder, label_file)

            with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse the dictionary using regex
                    match = re.search(r"'relative_within_patch_location': \[(\d+\.\d+), (\d+\.\d+)\]", line)
                    if match:
                        x_center = float(match.group(1))  # Normalized x_center
                        y_center = float(match.group(2))  # Normalized y_center

                        # Write to YOLO format: class_id, x_center, y_center, width, height
                        class_id = 0  # 'rock' class is class 0
                        outfile.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

            # print(f"Processed: {label_file}")

    print("Conversion to YOLOv8 format completed.")

# Data augmentation:

def calculate_mean_std(image_folder):
    """
    Calculate the mean and standard deviation of the dataset
    """
    pixel_sum = torch.zeros(3)
    pixel_squared_sum = torch.zeros(3)
    num_pixels = 0

    # Iterate through all files in the folder
    for file_name in os.listdir(image_folder):
        if file_name.endswith(".jpg"):
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

def geometric_augmentations(image, labels):
    """
    Apply geometric and image augmentations to the input image and labels.
    """
    augmented_images = []
    augmented_labels = []

    # Geometric transformations applied to both image and labels
    geometric_transforms = [
        ("vertical_flip", lambda img, lbl: (F.vflip(img), flip_y(lbl))),
        ("horizontal_flip", lambda img, lbl: (F.hflip(img), flip_x(lbl))),
    ]

    # Apply transformations sequentially without overlap
    for name, transform in geometric_transforms:
        aug_img, aug_lbl = transform(image, labels.clone())  # Ensure labels are cloned
        augmented_images.append(aug_img)
        augmented_labels.append(aug_lbl)

    return augmented_images, augmented_labels

# Helper functions for label transformations
def flip_x(labels):
    """Flip labels horizontally."""
    if labels.numel() == 0:
        return labels
    labels[:, 1] = 1 - labels[:, 1]  # Flip x_center
    return labels

def flip_y(labels):
    """Flip labels vertically."""
    if labels.numel() == 0:
        return labels
    labels[:, 2] = 1 - labels[:, 2]  # Flip y_center
    return labels

def brightning(image, labels, image_size=(640, 640)):
    """
    Apply brightness/contrast adjustments to the input image and labels:

    Args:
        image (PIL.Image): Input image.
        labels (torch.Tensor): YOLO bounding boxes [class, x_center, y_center, width, height].
        image_size (tuple): Size of the output image (height, width).

    Returns:
        List[Tuple[PIL.Image, torch.Tensor]]: List of (augmented_image, augmented_labels).
    """
    augmented_data = []
    original_labels = labels.clone()

    # 2. Brightness/Contrast Adjustments
    brightness_contrast_transform = T.ColorJitter(brightness=0.3, contrast=0.3)
    bright_image = brightness_contrast_transform(image)  # Original image used for light changes
    augmented_data.append((bright_image, original_labels))  # Labels unchanged

    return augmented_data

def obstruction(image, labels, image_size=(640, 640)):
    """
    Apply random obstruction to the input image and labels:

    Args:
        image (PIL.Image): Input image.
        labels (torch.Tensor): YOLO bounding boxes [class, x_center, y_center, width, height].
        image_size (tuple): Size of the output image (height, width).

    Returns:
        List[Tuple[PIL.Image, torch.Tensor]]: List of (augmented_image, augmented_labels).
    """
    augmented_data = []
    original_labels = labels.clone()

    # Random Obstruction with Fixed Size
    obstruction_image = image.copy()
    draw = ImageDraw.Draw(obstruction_image)
    for _ in range(random.randint(1, 3)):  # Add 1-3 fixed-size obstructions
        rect_x_min = random.randint(0, image_size[0] - 10)
        rect_y_min = random.randint(0, image_size[1] - 10)
        rect_x_max = rect_x_min + 10
        rect_y_max = rect_y_min + 10
        draw.rectangle([rect_x_min, rect_y_min, rect_x_max, rect_y_max], fill=(0, 0, 0))

    augmented_data.append((obstruction_image, original_labels))  # Labels unchanged

    return augmented_data

def denormalize(image_tensor, mean, std):
    """
    Reverse the normalization applied to the image tensor.
    """
    for t, m, s in zip(image_tensor, mean, std):
        t.mul_(s).add_(m)  # Reverse normalization: t = t * std + mean
    return image_tensor.clamp(0, 1)  # Ensure pixel values are in range [0, 1]

def save_augmented_data_batch(batch_images, batch_labels, batch_names, output_image_folder, output_label_folder, augmentation_names, mean=None, std=None):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    for original_image_name, augmented_images, augmented_labels in zip(batch_names, batch_images, batch_labels):
        base_name = os.path.splitext(original_image_name)[0]

        for idx, (image, labels) in enumerate(zip(augmented_images, augmented_labels)):
            # Generate file names
            augmented_image_name = f"{base_name}_aug_{idx + 1}.jpg"
            augmented_label_name = f"{base_name}_aug_{idx + 1}.txt"

            # Save image
            image_path = os.path.join(output_image_folder, augmented_image_name)
            image.save(image_path, format="JPEG")

            # Save labels
            label_path = os.path.join(output_label_folder, augmented_label_name)
            with open(label_path, "w") as f:
                for label in labels:
                    f.write(" ".join(map(str, label.tolist())) + "\n")  # Save transformed labels

class RockDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, label_folder, mean, std, augment=False):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
        self.label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.txt')])
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

        # # Apply augmentation if enabled
        # if self.augment:
        #     augmented_images, augmented_labels = transform_train_with_labels(image, labels_tensor)
        #     return augmented_images, augmented_labels

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