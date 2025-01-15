# Comparison of Yolov8 versions on large rocks detection

## Introduction

The Federal Office for Topography (swisstopo) undertakes the precise and labor-intensive task of manually annotating large rocks (over 5x5 meters) across Switzerland for the production of topographic maps. With advancements in automatic detection methods, swisstopo seeks to explore the feasibility of integrating such technologies to enhance efficiency and accuracy in their workflows.

This project investigates the application of the YOLOv8 object detection framework on swisstopo's large rock detection dataset. Specifically, we evaluate the performance of two YOLOv8 versions—'nano' and 'large'—trained on the dataset. The models are first trained on the provided training dataset and subsequently fine-tuned using the validation dataset to optimize their parameters. Finally, we analyze the models' performance on the test dataset by visualizing the results and computing a range of evaluation metrics.

One of the key challenges in this work lies in the complexity and variability of natural data. Factors such as varying lighting conditions, shadows, dense forest canopies, and the presence of man-made structures like houses introduce significant noise and ambiguity in the detection process. These challenges necessitate robust training methodologies and comprehensive evaluation to ensure the models can generalize effectively across diverse environmental conditions.

Two different pre-trained versions of YOLO were utilized for this analysis: the nano and large versions. While the large model is expected to produce more accurate predictions, our objective in evaluating both models is to assess whether the improved accuracy of the large model justifies its significantly higher computational cost compared to the nano version.

This study aims to provide a systematic evaluation of YOLOv8's capabilities for large rock detection, offering insights into the potential integration of modern machine learning techniques into geospatial annotation workflows.

## Large Rocks Dataset

The Large Rocks Dataset serves as the foundation for this project and is derived from high-resolution imagery and terrain models annotated by swisstopo. It offers a unique opportunity to compare traditional geospatial analysis techniques (e.g., local maximum detection, rugosity indices) with recent advancements in object detection models.

**Dataset Characteristics**
- Study Area: The dataset spans regions in Valais, Ticino, and Graubünden, which feature diverse topographical and environmental conditions.
- Geographic Splitting: Tiles are geographically divided into training and testing sets to ensure robust evaluation of model performance.

**Data Composition**
- Aerial Imagery: High-resolution RGB images (50 cm resolution) from swissIMAGE.
Provides detailed visual representation of the landscape.
- Digital Surface Model (DSM): Terrain elevation data at 50 cm resolution derived from LiDAR through swissSURFACE3D. Captures 3D surface details critical for identifying large rock structures.
- Hillshade Raster: Generated from DSM data using QGIS (Azimuth: 0°, Vertical Angle: 0°). Highlights surface topography through shaded relief, aiding in the visual detection of rock features.
- Annotations: Comprehensive point annotations for 2,625 large rocks provided by swisstopo experts. Forms the ground truth for training and evaluation of detection models.

**Dataset Access**
The dataset can be downloaded from the following link:
[Large Rocks Dataset - Download Here](https://enacshare.epfl.ch/bY2wS5TcA4CefGks7NtXg)

## Pipeline Description

The pipeline implemented for the large rock detection project is structured into distinct stages to effectively prepare, preprocess, and organize the dataset for training YOLOv8 models. Then load the best weights that we obtained after training both large and nano versions of YOLOv8 to visualise and evaluate their predictions on the test dataset.

### Data Preprocessing

**Combining DSM and RGB Data:**
A novel input format is created by replacing the red band of RGB images with the hillshade greyscale derived from the Digital Surface Model (DSM). To do so, the RGB and DSM images are first converted in .jpg using `convert_tif_to_jpg()`. This provides a new input representation for YOLOv8, combining topographic context with spectral data. This approach improves the model's ability to detect large rocks by incorporating terrain variations, reducing redundancy in overlapping RGB channels, and improving performance in shadowed areas where elevation data remains unaffected by lighting. However this introduces some trade-offs, like losing the spectral information from the red band, potentially affecting vegetation-based distinctions. These combined images becomes the dataset for the following steps.

**Dataset Splitting:**
The dataset is split into training and testing sets based on metadata provided in the `large_rock_dataset.json` file. This process is handled by the function `split_train_from_json()` which organizes the data into corresponding directories.
A validation set is generated by randomly selecting 10% of the training images (`create_validation_set_images()` and `create_validation_set_labels()`).

Labels are converted to YOLO format using `convert_labels_to_yolo_format()`. The yolo labels look like this:
`<class\_id> <x\_center> <y\_center>  <width> <height>`, with width and height beeing 30 pixels.

**Normalization and Augmentation:**
For the training dataset, a PyTorch-compatible dataset is created (class `RockDetectionDataset`) with normalization parameters calculated using `calculate_mean_std()`.

**Data augmentation is applied to enhance the model's robustness:**
- Geometric Transformations: Horizontal and vertical flips (`geometric_augmentations()`).
- Brightness Adjustment: Variations in image brightness and contrast (`brightning()`).
- Random Obstruction: Addition of 30x30 pixel black squares to simulate occlusions (`obstruction()`).

**Dataset Organization:**
All data is reorganized into a YOLO-compatible format in 'yolo_dataset' folder, with separate directories for images and labels under train, val, and test folders. With a `yolo_description.yaml` file explaining the structure.

### Model training

[Download the folder model from this Google Drive to get the both models weights](https://drive.google.com/drive/folders/1XsQa--gZmbfJvRCiTDBKF1DENv7QX57_?usp=sharing)

The 2 models were trained using the same hyperparameters. After extensive testing, the following hyperparameters were selected for the two models:

- **Epochs**: 150 – Number of complete passes through the training dataset.

- **Learning Rate**: 5 × 10⁻⁴ – Balances convergence speed and stability.

- **Weight Decay**: 5 × 10⁻⁴ – Regularization to prevent overfitting.

- **Batch Size**: 32 – Optimizes memory efficiency and gradient stability.

- **Data Enhancements**: augment, amp, and mosaic set to True:
  - Augment: Applies random transformations to improve generalization.
  - AMP (Automatic Mixed Precision): Combines 16-bit and 32-bit operations for faster training without sacrificing accuracy.
  - Mosaic: Combines four images during preprocessing, enhancing robustness to varying object sizes and spatial contexts.

Model Performance : A 50% accuracy threshold was selected for predictions on the validation and testing set.

### Results of the different versions

Using a small sample of the preprocessed images, we visualize the results alongside the ground truth for the validation set and the predictions for the testing set. Finally, we present various evaluation metrics generated by both models to compare their performance downloaded directly in the model folder.