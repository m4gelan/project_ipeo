from tifffile import tifffile 
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.transforms as T
import dataset as dt

def aug_pipeline_geom(dataset, mean, std, batch_size, output_image_folder, output_label_folder):

    # Batch processing loop
    batch_images, batch_labels, batch_names = [], [], []

    for idx in range(len(dataset)):
        # Load original image and labels
        original_image, original_labels = dataset[idx]
        original_image_name = dataset.image_files[idx]  # Get image name

        # Convert tensor to PIL image if needed
        if isinstance(original_image, torch.Tensor):
            original_image = dt.denormalize(original_image.clone(), mean, std)
            original_image = T.ToPILImage()(original_image)

        # Perform augmentations
        augmented_images, augmented_labels = dt.geometric_augmentations(original_image, original_labels)

        # Verify transformations (optional debugging)
        for i, (aug_img, aug_lbl) in enumerate(zip(augmented_images, augmented_labels)):
            print(f"Augmentation {i + 1} for {original_image_name}:")
            print("Labels:", aug_lbl.tolist())

        # Append to batch
        batch_images.append(augmented_images)
        batch_labels.append(augmented_labels)
        batch_names.append(original_image_name)

        # When batch is full, save the batch
        if len(batch_images) >= batch_size:
            dt.save_augmented_data_batch(
                batch_images, batch_labels, batch_names,
                output_image_folder, output_label_folder,
                augmentation_names=[f"aug_{i + 1}" for i in range(len(augmented_images))],
                mean=mean, std=std
            )
            # Reset batch
            batch_images, batch_labels, batch_names = [], [], []

    # Save remaining images in the last batch
    if len(batch_images) > 0:
        dt.save_augmented_data_batch(
            batch_images, batch_labels, batch_names,
            output_image_folder, output_label_folder,
            augmentation_names=[f"aug_{i + 1}" for i in range(len(augmented_images))],
            mean=mean, std=std
        )

def aug_pipeline_obstruction(dataset, mean, std, batch_size, output_image_folder, output_label_folder):

    # Batch processing loop
    batch_images, batch_labels, batch_names = [], [], []

    for idx in range(len(dataset)):
        # Load original image and labels
        original_image, original_labels = dataset[idx]
        original_image_name = dataset.image_files[idx]  # Get image name

        # Convert tensor to PIL image if needed
        if isinstance(original_image, torch.Tensor):
            original_image = dt.denormalize(original_image.clone(), mean, std)
            original_image = T.ToPILImage()(original_image)

        # Perform enhanced augmentations
        augmented_data = dt.obstruction(original_image, original_labels)

        # Append each augmented image and labels to the batch
        for aug_idx, (aug_image, aug_labels) in enumerate(augmented_data):
            batch_images.append([aug_image])  # Wrap in a list for batch saving
            batch_labels.append([aug_labels])
            batch_names.append(f"{original_image_name}_aug_{aug_idx + 1}")

        # When batch is full, save the batch
        if len(batch_images) >= batch_size:
            dt.save_augmented_data_batch(
                batch_images, batch_labels, batch_names,
                output_image_folder, output_label_folder,
                augmentation_names=[f"aug_{i + 1}" for i in range(len(batch_images))],
                mean=mean, std=std
            )
            # Reset batch
            batch_images, batch_labels, batch_names = [], [], []

    # Save remaining images in the last batch
    if len(batch_images) > 0:
        dt.save_augmented_data_batch(
            batch_images, batch_labels, batch_names,
            output_image_folder, output_label_folder,
            augmentation_names=[f"aug_{i + 1}" for i in range(len(batch_images))],
            mean=mean, std=std
        )

def aug_pipeline_brightning(dataset, mean, std, batch_size, output_image_folder, output_label_folder):
    
    # Batch processing loop
    batch_images, batch_labels, batch_names = [], [], []

    for idx in range(len(dataset)):
        # Load original image and labels
        original_image, original_labels = dataset[idx]
        original_image_name = dataset.image_files[idx]  # Get image name

        # Convert tensor to PIL image if needed
        if isinstance(original_image, torch.Tensor):
            original_image = dt.denormalize(original_image.clone(), mean, std)
            original_image = T.ToPILImage()(original_image)

        # Perform enhanced augmentations
        augmented_data = dt.brightning(original_image, original_labels)

        # Append each augmented image and labels to the batch
        for aug_idx, (aug_image, aug_labels) in enumerate(augmented_data):
            batch_images.append([aug_image])  # Wrap in a list for batch saving
            batch_labels.append([aug_labels])
            batch_names.append(f"{original_image_name}_aug_{aug_idx + 1}")

        # When batch is full, save the batch
        if len(batch_images) >= batch_size:
            dt.save_augmented_data_batch(
                batch_images, batch_labels, batch_names,
                output_image_folder, output_label_folder,
                augmentation_names=[f"aug_{i + 1}" for i in range(len(batch_images))],
                mean=mean, std=std
            )
            # Reset batch
            batch_images, batch_labels, batch_names = [], [], []

    # Save remaining images in the last batch
    if len(batch_images) > 0:
        dt.save_augmented_data_batch(
            batch_images, batch_labels, batch_names,
            output_image_folder, output_label_folder,
            augmentation_names=[f"aug_{i + 1}" for i in range(len(batch_images))],
            mean=mean, std=std
        )