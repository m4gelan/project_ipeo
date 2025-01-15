from ultralytics import YOLO
import os

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
    augment=True,
    mosaic=True,  # Enable mosaic augmentation
    amp=True,
    save_period=10
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
        conf=0.3,                       #confidence of at least 30%
        project=save_dir_pred,  # Set the output directory here
        imgsz=640             # Image size (ensure it matches your training)
        )
    
    return results