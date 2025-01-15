from ultralytics import YOLO
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt


def train_model(model_name,model,yaml_file_path, save_dir):
    model = YOLO(model)

    model.train(
    data=yaml_file_path,
    name=model_name,
    project=save_dir,
    optimizer = 'AdamW',
    epochs=200,
    imgsz=640,
    batch=32,
    lr0=0.0005,  # Lower learning rate
    weight_decay=0.0005,  # Add slight weight decay
    patience=15,  # Increase patience for early stopping
    augment=True,
    mosaic=True,  # Enable mosaic augmentation
    amp=True,
    save_period=5
    )
    return model

def predict_model(save_dir ,validation_set_path, save_dir_pred):
    os.makedirs(save_dir_pred, exist_ok=True)    
    model = YOLO(save_dir + '/weights/best.pt')
    results = model.predict(
        source= validation_set_path, 
        save=True,                     # Save prediction images
        save_txt=True,                 # Save predictions in YOLO format
        save_conf=True,
        conf=0.3, #confidence of at least 30%
        project=save_dir_pred,  # Set the output directory here
        imgsz=640             # Image size (ensure it matches your training)
        )
    
    return results

def load_yolo_labels(label_file):
    if not os.path.exists(label_file):
        return []
    with open(label_file, 'r') as f:
        lines = f.readlines()
    return [list(map(float, line.strip().split())) for line in lines]
def draw_boxes(image, boxes, color, label_type=""):
    h, w, _ = image.shape
    for box in boxes:
        x_center, y_center, width, height = box[1:5]  
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label_type}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def plot_model_predictions(image_folder, label_folder, model_inputs):
    """
    Plot the predictions of different models for a random selection of three images and compare their accuracy.

    Parameters:
    - image_folder (str): Path to the folder containing the images.
    - label_folder (str): Path to the folder containing the labels.
    - model_inputs (dict): Dictionary where keys are model names and values are tuples of model paths and prediction paths.
    """
    # Get and shuffle the image and label files
    image_files = sorted(os.listdir(image_folder))
    label_files = sorted(os.listdir(label_folder))
    
    # Ensure the image and label counts match
    assert len(image_files) == len(label_files), "Mismatch between image and label files!"
    
    # Randomly select three indices
    selected_indices = random.sample(range(len(image_files)), 3)
    
    for image_index in selected_indices:
        # Load the image
        image_path = os.path.join(image_folder, image_files[image_index])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load the ground truth labels
        label_path = os.path.join(label_folder, label_files[image_index])
        true_labels = load_yolo_labels(label_path)
        
        # Plot the image and predictions
        fig, ax = plt.subplots(1, len(model_inputs) + 1, figsize=(15, 5))
        ax[0].imshow(image)
        ax[0].set_title('Ground Truth')
        
        # Plot the ground truth labels
        draw_boxes(image, true_labels, color=(0, 255, 0), label_type="GT")  # Green for ground truth
        ax[0].imshow(image)
        
        # Plot predictions for each model
        for i, (model_name, (_, _, pred_path)) in enumerate(model_inputs.items()):
            pred_image = image.copy()
            pred_label_file = os.path.join(pred_path, 'predict/labels', label_files[image_index])
            pred_labels = load_yolo_labels(pred_label_file)
            draw_boxes(pred_image, pred_labels, color=(255, 0, 0), label_type="Pred")  # Red for predictions
            
            ax[i + 1].imshow(pred_image)
            ax[i + 1].set_title(model_name)
        
        plt.show()

#similar function as plot_model_predictions but without ground truth labels
def plot_test_predictions(image_folder, model_inputs, type):
    """
    Plot the predictions of different models for a random selection of test images.

    Parameters:
    - image_folder (str): Path to the folder containing the images.
    - model_inputs (dict): Dictionary where keys are model names and values are tuples of model paths and prediction paths.
    - type (str): Type of the test set (e.g., 'RGB', 'hillshade').
    """
    # Get the sorted list of image files
    image_files = sorted(os.listdir(image_folder))
    
    # Randomly select 3 images
    selected_indices = random.sample(range(len(image_files)), 3)
    
    for image_index in selected_indices:
        # Load the image
        image_path = os.path.join(image_folder, image_files[image_index])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a figure for the selected image
        fig, ax = plt.subplots(1, len(model_inputs), figsize=(15, 5))
        
        # Plot predictions for each model
        for i, (model_name, (_, _, pred_path)) in enumerate(model_inputs.items()):
            pred_image = image.copy()
            pred_label_file = os.path.join(
                f"{pred_path}_{type}",
                'predict/labels',
                image_files[image_index].replace('.jpg', '.txt')
            )
            pred_labels = load_yolo_labels(pred_label_file)
            
            if pred_labels:
                draw_boxes(pred_image, pred_labels, color=(255, 0, 0), label_type="Pred")  # Red for predictions
            else:
                cv2.putText(pred_image, "No predictions", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            ax[i].imshow(pred_image)
            ax[i].set_title(model_name)
        
        plt.show()
