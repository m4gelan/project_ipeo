from tifffile import tifffile 
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.transforms as T
import os
import shutil
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
    
    print('Geometric augmentation completed.')

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

    print('Obstruction augmentation completed.')


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

    print('Brightning augmentation completed.')

def organize_yolo_dataset(source_folder):
    # Create destination folder
    destination_folder = os.path.join('yolo_dataset')
    os.makedirs(destination_folder, exist_ok=True)

    # Define the new folder structure
    train_images = os.path.join(destination_folder, 'train_dataset', 'images')
    train_labels = os.path.join(destination_folder, 'train_dataset', 'labels')
    val_images = os.path.join(destination_folder, 'val_dataset', 'images')
    val_labels = os.path.join(destination_folder, 'val_dataset', 'labels')
    test_images = os.path.join(destination_folder, 'test_dataset', 'images')
    yaml_file = os.path.join(destination_folder, 'yolo_description.yaml')

    # Create all necessary folders
    for folder in [train_images, train_labels, val_images, val_labels, test_images]:
        os.makedirs(folder, exist_ok=True)

    # Define subfolders for each dataset type
    train_image_folders = ['train_images']
    train_label_folders = ['train_labels']
    val_image_folders = ['val_images']
    val_label_folders = ['val_labels']
    test_image_folders = ['test']

    def process_combined_folder(image_folders, label_folders, source_folder, dest_images, dest_labels):
        # Process images
        for folder in image_folders:
            folder_path = os.path.join(source_folder, folder)
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} does not exist, skipping folder.")
                continue

            for file_name in os.listdir(folder_path):
                try:
                    name, ext = os.path.splitext(file_name)
                    # Copy image
                    src = os.path.join(folder_path, file_name)
                    dest = os.path.join(dest_images, file_name)
                    if os.path.exists(src):
                        shutil.copy2(src, dest)

                except Exception as e:
                    print(f"Error processing image {file_name}: {str(e)}")

        # Process labels
        if label_folders and dest_labels:
            for folder in label_folders:
                folder_path = os.path.join(source_folder, folder)
                if not os.path.exists(folder_path):
                    print(f"Warning: {folder_path} does not exist, skipping folder.")
                    continue

                for file_name in os.listdir(folder_path):
                    try:
                        src = os.path.join(folder_path, file_name)
                        dest = os.path.join(dest_labels, file_name)
                        if os.path.exists(src):
                            shutil.copy2(src, dest)

                    except Exception as e:
                        print(f"Error processing label {file_name}: {str(e)}")

    # Process training dataset
    process_combined_folder(
        train_image_folders, train_label_folders,
        source_folder, train_images, train_labels
    )

    # Process validation dataset
    process_combined_folder(
        val_image_folders, val_label_folders,
        source_folder, val_images, val_labels
    )

    # Process test dataset (images only)
    process_combined_folder(
        test_image_folders, None,
        source_folder, test_images, None
    )

    # Create the YOLO description YAML file
    with open(yaml_file, 'w') as f:
        f.write(f"train: {os.path.join(destination_folder, 'train_dataset')}\n")
        f.write(f"val: {os.path.join(destination_folder, 'val_dataset')}\n")
        f.write("nc: 1  # Number of classes\n")
        f.write("names: ['rock']  # Class names\n")

    print(f"YOLO dataset organized in '{destination_folder}'")


# def organize_yolo_dataset(source_folder_rgb, source_folder_hillshade):
#     # Create destination folder
#     destination_folder = os.path.join('yolo_dataset')
#     os.makedirs(destination_folder, exist_ok=True)

#     # Define the new folder structure
#     train_images = os.path.join(destination_folder, 'train_dataset', 'images')
#     train_labels = os.path.join(destination_folder, 'train_dataset', 'labels')
#     val_images = os.path.join(destination_folder, 'val_dataset', 'images')
#     val_labels = os.path.join(destination_folder, 'val_dataset', 'labels')
#     test_images = os.path.join(destination_folder, 'test_dataset', 'images')
#     yaml_file = os.path.join(destination_folder, 'yolo_description.yaml')

#     # Create all necessary folders
#     for folder in [train_images, train_labels, val_images, val_labels, test_images]:
#         os.makedirs(folder, exist_ok=True)

#     # Define subfolders for each dataset type
#     train_image_folders = [
#         'train_images',
#         'augmented_train_images_geom',
#         'augmented_train_images_brightning',
#         'augmented_train_images_obstruction'
#     ]

#     train_label_folders = [
#         'yolo_train_labels',
#         'augmented_train_labels_geom',
#         'augmented_train_labels_brightning',
#         'augmented_train_labels_obstruction'
#     ]

#     val_image_folders = ['val_images']
#     val_label_folders = ['yolo_val_labels']

#     test_image_folders = ['test']

#     # Define suffix mapping for augmented images
#     suffix_map = {
#         'augmented_train_images_geom': 'geom',
#         'augmented_train_images_brightning': 'bright',
#         'augmented_train_images_obstruction': 'obst',
#         'augmented_train_labels_geom': 'geom',
#         'augmented_train_labels_brightning': 'bright',
#         'augmented_train_labels_obstruction': 'obst'
#     }

#     def process_images_and_labels(image_folders, label_folders, source_folder_rgb, source_folder_hillshade, 
#                                 dest_images, dest_labels, suffix_map):
#         # Process images
#         for folder in image_folders:
#             rgb_path = os.path.join(source_folder_rgb, folder)
#             hillshade_path = os.path.join(source_folder_hillshade, folder)

#             if not os.path.exists(rgb_path):
#                 print(f"Warning: {rgb_path} does not exist, skipping folder.")
#                 continue

#             for file_name in os.listdir(rgb_path):
#                 try:
#                     name, ext = os.path.splitext(file_name)
#                     suffix = suffix_map.get(folder, '')
                    
#                     # Create base name for both image and label
#                     base_name = f"{name}_{suffix}" if suffix else name
                    
#                     # Copy RGB image
#                     rgb_src = os.path.join(rgb_path, file_name)
#                     rgb_dest = os.path.join(dest_images, f"{base_name}_rgb{ext}")
#                     if os.path.exists(rgb_src):
#                         shutil.copy2(rgb_src, rgb_dest)
                    
#                     # Copy hillshade image
#                     hillshade_src = os.path.join(hillshade_path, file_name)
#                     hillshade_dest = os.path.join(dest_images, f"{base_name}_hillshade{ext}")
#                     if os.path.exists(hillshade_src):
#                         shutil.copy2(hillshade_src, hillshade_dest)

#                 except Exception as e:
#                     print(f"Error processing image {file_name}: {str(e)}")

#         # Process labels if label folders are provided
#         if label_folders and dest_labels:
#             for folder in label_folders:
#                 label_path = os.path.join(source_folder_rgb, folder)
                
#                 if not os.path.exists(label_path):
#                     print(f"Warning: {label_path} does not exist, skipping folder.")
#                     continue

#                 for file_name in os.listdir(label_path):
#                     try:
#                         name, ext = os.path.splitext(file_name)
#                         suffix = suffix_map.get(folder, '')
#                         base_name = f"{name}_{suffix}" if suffix else name
                        
#                         # Copy label file
#                         label_src = os.path.join(label_path, file_name)
#                         label_dest = os.path.join(dest_labels, f"{base_name}{ext}")
#                         if os.path.exists(label_src):
#                             shutil.copy2(label_src, label_dest)

#                     except Exception as e:
#                         print(f"Error processing label {file_name}: {str(e)}")

#     # Process training dataset
#     process_images_and_labels(
#         train_image_folders, train_label_folders,
#         source_folder_rgb, source_folder_hillshade,
#         train_images, train_labels, suffix_map
#     )

#     # Process validation dataset
#     process_images_and_labels(
#         val_image_folders, val_label_folders,
#         source_folder_rgb, source_folder_hillshade,
#         val_images, val_labels, suffix_map
#     )

#     # Process test dataset (images only)
#     process_images_and_labels(
#         test_image_folders, None,
#         source_folder_rgb, source_folder_hillshade,
#         test_images, None, suffix_map
#     )

#     # Create the YOLO description YAML file
#     with open(yaml_file, 'w') as f:
#         f.write(f"train: {os.path.join(destination_folder, 'train_dataset')}\n")
#         f.write(f"val: {os.path.join(destination_folder, 'val_dataset')}\n")
#         f.write("nc: 1  # Number of classes\n")
#         f.write("names: ['rock']  # Class names\n")

#     print(f"YOLO dataset organized in '{destination_folder}'")