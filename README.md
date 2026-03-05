# AI_FOR_SUSTAINABILITY
Land Use Classification of Delhi-NCR using Satellite Imagery and ResNet-18 for AI-driven sustainability.

PROJECT OVERVIEW

This repository contains a complete end-to-end pipeline for classifying satellite imagery into five distinct land-use categories to support urban sustainability research. Using a combination of geospatial data engineering and deep learning, this project achieves high-accuracy mapping of the Delhi-NCR region.

KEY METICS:

Accuracy: 93.23%

Weighted F1-Score: 0.93

Final Dataset Size: 8,015 labeled image patches

TECHNICAL WORKFLOW 

This Project Mainly consists of two part:

1.Geospatial Data Engineering (Q1 & Q2)

Spatial Filtering: Utilized GeoPandas and Shapely to perform a point-in-polygon check, filtering raw satellite images to ensure they fall within the official Delhi-NCR boundary.

Label Construction: Developed a custom extraction engine using rasterio. By applying an Affine Transformation, I mapped geographic coordinates to pixel indices in the ESA WorldCover 2021 .tif file.

Dominant Class Assignment: Extracted 128x128 patches and assigned labels using the statistical mode to determine the primary land use for each image.

2. Deep Learning Architecture (Q3)

Model: Utilized ResNet-18 with Transfer Learning.

Optimization: Employed the Adam optimizer and Cross-Entropy loss over 5 epochs.

Stratification: Implemented a stratified 60/40 train-test split to maintain class distribution across subsets, ensuring a fair evaluation of minority classes like 'Water' and 'Others'.

PERFORMANCE 

- Accuracy : 93.23%

- Weighted F1-Score : 0.93

REPOSITORY STRUCTURE

data/: GeoJSON boundary files.

scripts/: Python implementations for spatial filtering, labeling, and CNN training.

visualizations/: Plots for class distribution and the final Confusion Matrix.

land_use_dataset.csv: The final processed dataset used for training.

NOTE : BEFORE TRY TO RUN THE CODE CHECK FOR THE PATHS(Please update the path variables at the top of the scripts to match your local data storage location before running."). YOU CAN DIRECTLY DOWNLOAD ALL DATA FROM LINK GIVEN ON AI for Sustainability support doc ATTACHED IN DATA FOLDER.

CREDITS

INDEPENDENT WORK: All geospatial logic, spatial filtering (Q1), and automated labeling workflows (Q2) were developed independently by me by learning the required libraries for the operations from various internet source mainly GeeksforGeeks.

AI MENTORSHIP: The deep learning implementation and PyTorch training loop (Q3) were developed with technical guidance from an AI mentor (Gemini). While the conceptual architecture and evaluation metrics are understood, AI was used to ensure industry-standard coding practices and syntax accuracy


