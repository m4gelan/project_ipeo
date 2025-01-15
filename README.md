# Comparison of Yolov8 versions on large rocks detection

## Introduction

The Federal Office for Topography (swisstopo) undertakes the precise and labor-intensive task of manually annotating large rocks (over 5x5 meters) across Switzerland for the production of topographic maps. With advancements in automatic detection methods, swisstopo seeks to explore the feasibility of integrating such technologies to enhance efficiency and accuracy in their workflows.

This project investigates the application of the YOLOv8 object detection framework on swisstopo's large rock detection dataset. Specifically, we evaluate the performance of two YOLOv8 versions—'nano' and 'large'—trained on the dataset. The models are first trained on the provided training dataset and subsequently fine-tuned using the validation dataset to optimize their parameters. Finally, we analyze the models' performance on the test dataset by visualizing the results and computing a range of evaluation metrics.

One of the key challenges in this work lies in the complexity and variability of natural data. Factors such as varying lighting conditions, shadows, dense forest canopies, and the presence of man-made structures like houses introduce significant noise and ambiguity in the detection process. These challenges necessitate robust training methodologies and comprehensive evaluation to ensure the models can generalize effectively across diverse environmental conditions.

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

## Code: 'Inference.py'

### 1. Data Preprocessing

combination : You could compare the usage of one source of data (e.g., only DSM or only RGB) or try to combine them. 
Split 
Val
Tensor
Augemntation
Yolo input ready

### 2. Model training

How we did it. Here is the diferent trained models : ...........

### 3. Results of the different versions

Taking a small sample of the preprocessed images we visualise the results and evaluation metrics.
