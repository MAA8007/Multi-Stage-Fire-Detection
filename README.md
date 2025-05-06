# Early Prediction and Area Estimation of Stubble Burning in Punjab

This repository contains the code and resources for a project focused on the early prediction of stubble burning events in Punjab and a planned extension for estimating the spatial extent of burned areas using deep learning and multi-modal satellite imagery.

## Project Overview

Stubble burning, the post-harvest practice of setting fire to crop residue, is a major contributor to severe air pollution and seasonal smog in the Punjab region. This project initially explored stubble burning *detection* but has pivoted to the more impactful challenge of **early prediction**, aiming to forecast where burning might occur (hence an Early Warning System) [user-provided text block on Stubble Burning Prediction]. This shift necessitated significant changes in our data collection and modeling strategies.

The project now aims to:
1.  **Early Predict:** Forecast the likelihood of stubble burning on a specific agricultural field up to 10 days in advance using pre-burn temporal patterns from Sentinel-2 (S2) optical and Sentinel-1 (SAR) radar data [deliverable4.ipynb].
2.  **Estimate Area :** Segment and quantify the exact burned area after an event has occurred, using high-resolution Sentinel-2 imagery and a SegFormer-based segmentation model, adapted through transfer learning from a forest fire segmentation task.

We leverage the complementary strengths of S2 and SAR data for prediction, particularly SAR's ability to penetrate cloud cover [deliverable4.ipynb]. The core of our prediction model is a Dual-Input Transformer architecture, which processes S2 and SAR data separately, significantly improving upon baseline methods [deliverable4.ipynb].

## Project Pipeline

The project follows a comprehensive pipeline from data acquisition to model deployment and future enhancements:

### 1. Data Acquisition and Dataset Creation (using Google Earth Engine - GEE)

This project utilizes datasets created using Google Earth Engine (GEE). It has been designed to facilitate the **prediction** of stubble burning in the agricultural regions of Punjab. The process leverages satellite imagery and active fire observations spanning several years (2019-2024) to build a comprehensive collection of data representing both areas affected by burning and those that remained unburned [deliverable_2.ipynb].

* **Geographical Scope & Timeframe:** The analysis focuses on a large area encompassing key agricultural zones within Punjab. We established a consistent timeframe for analysis each year, specifically the burning season which typically occurs between September and December. For the prediction task, data collection for each point starts from June 1st [deliverable_2.ipynb].
* **Cropland Masking:** To ensure focus on agricultural activities, a pre-existing map identifying cropland areas was incorporated, effectively masking out non-agricultural land [deliverable_2.ipynb].
* **Data Sources:**
    * **Sentinel-2 (S2) MSI:** Optical data providing 10 spectral bands (B2-B12) and 7 derived indices (NDVI, NDWI, NDRE, MSAVI, SAVI, NBR, BAIS2). S2 samples with >40% cloud cover were excluded as they would significantly distort metrics [deliverable_2.ipynb, deliverable4.ipynb].
    * **Sentinel-1 (SAR):** C-band Synthetic Aperture Radar data, providing VV and VH backscatter, ratios (RFDI, RVI), and texture features. SAR is crucial due to its cloud penetration capabilities. This was utlized in the 2nd improvement [deliverable4.ipynb].
* **Point Labeling & Feature Extraction (Prediction Task):** The core of the dataset generation process centered around identifying locations indicative of stubble burning. This was achieved by analyzing active fire data obtained from satellites. When these fire locations coincided with our defined cropland areas, they were labeled as 'burned' sample points. To provide a comparative dataset, 'unburned' sample points were selected as random locations within the cropland areas that were a certain distance away from any detected fires. Around each of these identified burned and unburned locations, small virtual buffer zones served as areas from which to extract relevant information from S2 imagery. From this optical imagery, key indicators such as the Normalized Difference Vegetation Index (NDVI) and the Normalized Burn Ratio (NBR) were calculated [deliverable_2.ipynb].
* **Output:** The culmination of this process was the creation of a structured, tabular dataset, making it suitable for training DL models to distinguish between (or predict) burned and unburned agricultural fields [deliverable_2.ipynb]. For the prediction task, a balanced dataset of 1051 unique burned and 1051 unburned field-year identifiers (`item_id`) was used.
* **Notebook:** `deliverable_2.ipynb` (formerly `eda.ipynb`) details this GEE process and initial EDA.

### 2. Data Preprocessing for Early Prediction

The shift from detection to prediction required significant changes to our data collection process []:
* **Temporal Cutoff:**
    * *Burned Fields:* Time series data collected from June 1st up to 10 days *before* the confirmed burn date.
    * *Unburned Fields:* Time series data collected from June 1st up to September 15th of the respective year. This ensures the model learns from pre-burn indicators only [deliverable4.ipynb].
* **Temporal Splitting:** A strict year-based temporal split was utilized to ensure the model generalizes well and is not memorizing year-specific patterns:
    * Training: 2019, 2020, 2021, 2022 (1410 fields)
    * Validation: 2023 (350 fields)
    * Testing: 2024 (342 fields) [deliverable4.ipynb].
* **Sequence Preparation:**
    * Data sorted chronologically. Missing values imputed using series mean. Features scaled using `StandardScaler` (fitted only on training). Sequences padded/truncated: S2 to 30 steps, SAR to 60 steps.

### 3. Modeling for Early Prediction

* **Baseline Model (S2 Transformer - Original Detection Task):**
    * An initial Transformer model using only normalized Sentinel-2 sequences for *burn detection*.
    * **Notebook:** `deliverable_3.ipynb` (formerly `deliverable_3 (4).ipynb`).
* **Random Forest Baseline (Prediction Task):**
    * Trained on static statistical features. Compared against the Transformer to show its effectiveness. Performance: 50% Accuracy, 0.00 F1-Score.
* **Dual-Input Transformer (Prediction Task - Main Model):**
    * Instead of a vanilla, single-input transformer, a Dual-Input Transformer is used, utilizing both SAR and S2 data. This approach was chosen because SAR and S2 satellites do not orbit over a single location at the same time, and fusing datasets would lead to discarding many SAR samples due to S2 cloud cover issues (>40% cloud cover samples were excluded).
    * Architecture: Parallel S2/SAR branches (Embedding -> Positional Encoding -> Transformer Encoder -> Pooling) -> Concatenation -> Dense Layers.
    * Addressed initial drastic overfitting (test accuracy ~70%) by incorporating dropout and regularization, leading to markedly better results.
    * Performance (2023 Test Set): ≈99.14% Accuracy, ≈0.9915 F1-Score.
    * **Notebook:** `deliverable4.ipynb` (formerly `deliverable4 (3).ipynb`).

### 4. Burned Area Segmentation (Future Work)

To complement the prediction of *likelihood*, this phase aims to estimate the *spatial extent* of burned areas post-event using image segmentation. Image segmentation is crucial for precise spatial quantification, detailed mapping, change detection, and targeted analysis of environmental events like fires.
* **Model:** SegFormer architecture (as prototyped with a U-Net/ResNet34 previously). The current focus is on segmenting forest fires using the CEMS Wildfires dataset, with plans to adapt this for stubble burning.
* **Input Enhancement:** Incorporates spectral indices (NBR and NDVI) as additional input channels alongside raw Sentinel-2 bands to better distinguish burned areas from spectrally similar features.
* **Training Strategy (for Stubble Burning):** **Transfer Learning / Fine-tuning**. The model pre-trained on the CEMS forest fire dataset (which learns features for identifying burned biomass from S2 data) will be fine-tuned on a new dataset specifically containing examples of stubble burning events. This approach is data-efficient, time-efficient, and often leads to better performance.
* **Dataset (Initial Task):** CEMS Wildfires dataset (`links-ads/wildfires-cems`) for forest fire segmentation.
* **Notebook (Forest Fire Segmentation & Transfer Learning Plan):** `final_improvement.ipynb`.

## Repository Structure
