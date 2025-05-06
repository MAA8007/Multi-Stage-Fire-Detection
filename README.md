# Early Prediction and Area Estimation of Stubble Burning in Punjab

This repository contains the code and resources for a project focused on the early prediction of stubble burning events in Punjab (Pakistan/India) and a planned extension for estimating the spatial extent of burned areas using deep learning and multi-modal satellite imagery.

## Project Overview

Stubble burning, the post-harvest practice of setting fire to crop residue, is a major contributor to severe air pollution and seasonal smog in the Punjab region. This project aims to develop a system that can:
1.  **Early Predict:** Forecast the likelihood of stubble burning on a specific agricultural field up to 10 days in advance using pre-burn temporal patterns from satellite data.
2.  **Estimate Area (Future Work):** Segment and quantify the exact burned area after an event has occurred, using high-resolution satellite imagery.

We leverage Sentinel-2 (S2) optical imagery and Sentinel-1 (SAR) radar data, recognizing their complementary strengths, particularly SAR's ability to see through clouds. The core of our prediction model is a Dual-Input Transformer architecture, which has shown significant improvement over baseline methods.

## Project Pipeline

The project follows a comprehensive pipeline from data acquisition to model deployment and future enhancements:

### 1. Data Acquisition and Dataset Creation (using Google Earth Engine - GEE)

* **Study Area:** Key agricultural zones in Punjab, Pakistan/India.
* **Timeframe:** 2019-2024, focusing on the typical burning season (September-December), with data collection for prediction starting from June 1st each year.
* **Data Sources:**
    * **Sentinel-2 (S2) MSI:** Optical data providing 10 spectral bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12) and 7 derived indices (NDVI, NDWI, NDRE, MSAVI, SAVI, NBR, BAIS2). Images with >40% cloud cover are excluded.
    * **Sentinel-1 (SAR):** C-band Synthetic Aperture Radar data, providing VV and VH backscatter polarizations, derived ratios (RFDI, RVI), and texture features (VH\_contrast, VH\_entropy, VV\_contrast).
* **Labeling (Prediction Task):**
    * Burned points (`burn_status=1`): Identified where active fire locations (from MODIS/VIIRS) coincided with cropland areas during the burn season.
    * Unburned points (`burn_status=0`): Randomly sampled within cropland areas, distanced from detected fires.
    * A balanced dataset of 1051 unique burned and 1051 unburned field-year identifiers (`item_id`) common to both S2 and SAR was created after filtering for the prediction task.
* **Notebook:** `eda.ipynb` details this process.

### 2. Data Preprocessing for Early Prediction

* **Temporal Cutoff:**
    * *Burned Fields:* Time series data from June 1st up to 10 days *before* the confirmed burn date.
    * *Unburned Fields:* Time series data from June 1st up to September 15th of the respective year. This prevents the model from learning post-harvest patterns not available in a forecasting scenario.
* **Temporal Splitting:** To ensure robust evaluation and prevent data leakage from future years, the data is split strictly by year:
    * Training: 2019, 2020, 2021, 2022 (1410 fields)
    * Validation: 2024 (342 fields)
    * Testing: 2023 (350 fields)
* **Sequence Preparation:**
    * Data is sorted chronologically for each `item_id`.
    * Missing values within each time series are imputed using the series mean.
    * Features are scaled using `StandardScaler` (fitted only on the training set).
    * Sequences are padded/truncated to fixed lengths: 30 steps for S2, 60 for SAR.

### 3. Modeling for Early Prediction

* **Baseline Model (S2 Transformer - Detection Task):**
    * An initial Transformer model using only normalized Sentinel-2 sequences (15 steps) for *burn detection*.
    * Achieved high validation F1 (~0.99), demonstrating Transformer viability.
    * **Notebook:** `deliverable_3 (4).ipynb`
* **Random Forest Baseline (Prediction Task):**
    * Trained on 96 static statistical features (mean, std, min, max per S2/SAR feature).
    * Performance: 50% Accuracy, 0.00 F1-Score on the 2023 test set, indicating insufficiency of static features for prediction.
* **Dual-Input Transformer (Prediction Task - Main Model):**
    * Processes S2 and SAR time series in parallel branches.
    * Architecture: Embedding -> Positional Encoding -> Transformer Encoder Block(s) (1 layer, 2 heads, 128 FF units) -> Global Average Pooling -> Concatenation -> Dense Layers for classification.
    * Includes L2 regularization and Dropout (0.4) to mitigate overfitting.
    * Performance (2023 Test Set): ≈99.14% Accuracy, ≈0.9915 F1-Score.
    * **Notebook:** `deliverable4 (3).ipynb`

### 4. Burned Area Segmentation (Future Work)

To complement the prediction of *likelihood*, this phase aims to estimate the *spatial extent* of burned areas after an event.
* **Model:** U-Net architecture with a ResNet34 backbone.
* **Input:** Post-burn Sentinel-2 imagery patches.
* **Training:** Will involve transfer learning, fine-tuning a pre-trained ResNet34 encoder on a custom dataset of S2 burn scar images and ground-truth masks.
* **Evaluation:** Intersection over Union (IoU) metric.
* **Notebook (Prototype):** `Untitled-2 _1_-3.ipynb`


## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone [Your GitHub Link Here]
    cd [repository-name]
    ```
2.  **Set up environment:** (Provide instructions, e.g., conda environment.yml or requirements.txt)
3.  **Data:** Download preprocessed data from [link if available] or run GEE scripts in `data/` (if provided). Place CSVs in a directory and update `DATA_DIR` in `deliverable4 (3).ipynb`.
4.  **Run Notebooks:**
    * Explore data creation in `eda.ipynb`.
    * See the baseline detection model in `deliverable_3 (4).ipynb`.
    * Run `deliverable4 (3).ipynb` for the main prediction model training and evaluation.
    * Review the segmentation prototype in `Untitled-2 _1_-3.ipynb`.

## Key Findings & Contributions

* **Early Prediction Feasibility:** Demonstrated that stubble burning can be predicted with high accuracy (≈99%) up to 10 days in advance using multi-modal satellite time series.
* **Dual-Input Transformer Efficacy:** Showcased the superiority of a sequence-based Dual-Input Transformer over static feature models (Random Forest) for capturing predictive temporal patterns.
* **Multi-Modal Advantage:** Integrated S2 optical and SAR radar data, leveraging SAR's cloud-penetrating capabilities for more robust analysis.
* **Temporal Validation:** Employed a strict temporal split (training on 2019-2022, testing on 2023, validating on 2024) to ensure model generalization to unseen years.
* **Comprehensive Pipeline:** Outlined an end-to-end approach from GEE-based dataset creation to prediction, with a clear path towards future integration of burned area segmentation for spatial extent estimation.

## Future Work

* Fully implement and evaluate the U-Net based burned area segmentation model with a ResNet34 backbone using transfer learning.
* Create and curate a high-quality labeled dataset for burned area segmentation specific to the Punjab region.
* Investigate the optimal prediction window (currently 10 days pre-burn).
* Explore incorporating additional data sources (e.g., meteorological data, soil moisture) to potentially enhance prediction accuracy and lead time.
* Refine model architectures and hyperparameter tuning.

## Acknowledgements
Muhammad Ashhad Ali (Project Partner)
ashhadali7703@gmail.com

## Contact
arsalanmalik900@outlook.com
