# Comparison of Yolov8 versions on large rocks detection

## Introduction

The Federal Office for Topography swisstopo proceeds to the manual annotations of all large rocks (over 5x5m) in Switzerland to produce topographic maps. They are curious to observe what could be done with recent automatic methods. In this project we provide them with an insight on different Yolov8 versions performance on their large rock detection dataset. 
Yolov8 ...

## Large Rocks Dataset

This dataset is based on their annotations on high-resolution RGB images and on the digital sur face model (DSM). Another possibility is to explore the differences between standard machine learning approaches (detecting local maximums, rugosity indices, etc.) and recent object detection models.
 
 
 Data ThestudyareaisspreadacrossValais, Ticino andGraubunden. Thetilesare geographically split
 into training and testing. The dataset includes :
 • Aerial images at a 50cm resolution with RGB bands (swissIMAGE)
 • Digital surface model (DSM) at 50cm resolution based on LiDAR data (swissSURFACE3D)
 • Hillshade raster tiles derived from the DSM data, generated with QGIS with the hillshade func
tion (Azimut 0, Vertical angle 0)
 • Comprehensive points annotations of 2’625 large rocks from swisstopo annotators.

The given dataset for this project need to be downloaded here : https://enacshare.epfl.ch/bY2wS5TcA4CefGks7NtXg

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
