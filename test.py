import torch
import torchvision.transforms as T
from PIL import Image
import os
import random
import numpy as np
from tqdm import tqdm

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

def transform_train_with_labels(image, label):
 
    # Geometric transformation applied to both image and label
    geometric_transforms =[T.RandomVerticalFlip(p=1),
                           T.RandomHorizontalFlip(p=1),
                           T.RandomRotation(degrees = (90,90)),
                           T.RandomRotation(degrees = (180,180)),
                           T.RandomRotation(degrees = (270,270))]
    
    geometric_transforms_names = ["vertical_flip", "horizontal_flip", "rotation_90", "rotation_180°", "rotation_270"]
    
    # Randomly select 3 transformations
    selected_transforms = random.sample(list(zip(geometric_transforms, geometric_transforms_names)), 3)

    # Extract the selected transformations and their names
    geometric_transforms, geometric_transforms_names = zip(*selected_transforms)
    geometric_transforms = list(geometric_transforms)
    geometric_transforms_names = list(geometric_transforms_names)

    new_images = [transform(image) for transform in geometric_transforms]
    new_labels = [transform(label) for transform in geometric_transforms]
    
    # Other transformation applied only to the image
    if random.random() < 0.5:
      other_transforms = [T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))]
      other_transforms_name = ["gaussian_blur"]
    else:
      other_transforms = []
      other_transforms_name = []
    
    new_images = new_images +  [transform(image) for transform in other_transforms]
    new_labels = new_labels + [label for _ in other_transforms]           
    return new_images, new_labels, geometric_transforms_names , other_transforms_name

def save_to_jpg(tensor, file_path):
    Image.fromarray((tensor.numpy()*255).astype(np.uint8)).save(file_path, format='JPEG')

class RockDetectionDataset(Dataset):
    def __init__(self, image_folder, label_folder, mean, std, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
        self.label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.txt')])
        self.transform = transform
        self.normalize = T.Normalize(mean=mean, std=std)

        assert len(self.image_files) == len(self.label_files)

    def __len__(self):
        return len(self.image_files)
    
    def load_patches(self, training = False):
        if self.loaded_ant_img is None:
            self.loaded_ant_img = [self.get_tensor_image(image_path) for image_path in tqdm(self.ant_dict.values(), desc="Loading Images")]
        if self.loaded_masks is None:
            self.loaded_masks = [self.get_tensor_mask(mask_path) for mask_path in tqdm(self.mask_dict.values(), desc="Loading Masks")]
            
        #Loop generating the augmented data
        if self.augmentation and training:
            augmented_patches_rgb = []
            augmented_patches_gt = []

            for image, label in tqdm(zip(self.loaded_ant_img, self.loaded_masks), desc ='Loading Augmentation'):
                new_images, new_labels, _, _ = transform_train_with_labels(image, label)
                augmented_patches_rgb.extend(new_images) 
                augmented_patches_gt.extend(new_labels) 

            #Add the augmented data to the original one
            self.loaded_ant_img.extend(augmented_patches_rgb)
            self.loaded_masks.extend(augmented_patches_gt)
            
            print(f"Total images (with augmentation):{len(self.loaded_ant_img)}, Total label (with augmentation) : {len(self.loaded_masks)} ")

        self.means, self.stds = self.calculate_mean_std()

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