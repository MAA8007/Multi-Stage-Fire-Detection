# Early Prediction and Area Estimation of Stubble Burning in Punjab

## Project Overview

Stubble burning, the post-harvest practice of setting fire to crop residue, is a major contributor to severe air pollution and seasonal smog in the Punjab region. This project initially explored stubble burning detection. However, the primary focus has pivoted to the more impactful challenge of **early prediction**, aiming to forecast where burning might occur, thereby enabling an Early Warning System. This shift necessitated significant changes in our data collection and modeling strategies.

The project now aims to:
1.  **Early Predict:** Forecast the likelihood of stubble burning on a specific agricultural field up to 10 days in advance using pre-burn temporal patterns from Sentinel-2 (S2) optical and Sentinel-1 (SAR) radar data.
2.  **Estimate Area (Future Work):** Segment and quantify the exact burned area after an event has occurred, using high-resolution Sentinel-2 imagery and an advanced segmentation model, adapted through transfer learning.

We leverage the complementary strengths of S2 and SAR data for prediction, particularly SAR's ability to penetrate cloud cover, which often limits optical data quality. The core of our prediction model is a Dual-Input Transformer architecture, which processes S2 and SAR data separately to account for their asynchronous acquisition and to avoid data loss from merging. This approach has shown significant improvement over baseline methods.

## Project Pipeline

The project follows a comprehensive pipeline from data acquisition to model deployment and future enhancements:

### 1. Data Acquisition and Dataset Creation (using Google Earth Engine - GEE)

This project utilizes datasets created using Google Earth Engine. It has been designed to facilitate the prediction of stubble burning in the agricultural regions of Punjab. The process leverages satellite imagery and active fire observations spanning several years (2019-2024) to build a comprehensive collection of data representing both areas affected by burning and those that remained unburned.

The initial phase involved defining the geographical scope of our analysis, focusing on a large area encompassing key agricultural zones within Punjab. We also established a consistent timeframe for analysis each year, specifically the burning season which typically occurs between September and December. To ensure our focus remained on agricultural activities, we incorporated a pre-existing map identifying cropland areas, effectively masking out non-agricultural land.

The core of the dataset generation process centered around identifying locations indicative of stubble burning. This was achieved by analyzing active fire data obtained from satellites. When these fire locations coincided with our defined cropland areas, they were labeled as 'burned' sample points. To provide a comparative dataset for training a model, we also identified 'unburned' sample points. These were selected as random locations within the cropland areas that were a certain distance away from any detected fires, ensuring they likely represented undisturbed farmland.

Around each of these identified burned and unburned locations, we created small virtual buffer zones. These zones served as areas from which to extract relevant information from satellite imagery. For each burning season, we processed optical imagery (like that from Sentinel-2, which captures visible and infrared light). From the optical imagery, we calculated key indicators such as the Normalized Difference Vegetation Index (NDVI) and the Normalized Burn Ratio (NBR), which are sensitive to vegetation health and burn severity.

The culmination of this process was the creation of a structured dataset. This information is organized in a tabular format, making it suitable for training DL models.

* **Data Sources used:**
    * **Sentinel-2 (S2) MSI:** Optical data providing 10 spectral bands and 7 derived indices. S2 samples with >40% cloud cover were excluded.
    * **Sentinel-1 (SAR):** Radar data providing VV and VH backscatter, ratios, and texture features.
* **Output:** For the prediction task, a balanced dataset of 1051 unique burned and 1051 unburned field-year identifiers (`item_id`) was used.
* **Notebook:** `deliverable_2.ipynb` details this GEE process and initial EDA.

### 2. Data Preprocessing for Early Prediction

In order to fully understand the improvements made for the prediction task, it is imperative to note the significant changes to our data collection process. Initially, for the baseline (detection task), we were just using Sentinel-2 data from a fixed date to another fixed date. For the prediction improvement, we have retrieved, for each location, both SAR data and Sentinel-2 data.

* **Temporal Cutoff:**
    * *Burned Fields:* Time series data collected from June 1st up to 10 days *before* the confirmed burn date.
    * *Unburned Fields:* Time series data collected from June 1st up to September 15th of the respective year. This ensures the model learns from pre-burn indicators only.
* **Temporal Splitting:** A strict year-based temporal split was utilized to ensure the model generalizes well:
    * Training: 2019, 2020, 2021, 2022 (1410 fields)
    * Validation: 2023 (350 fields)
    * Testing: 2024 (342 fields) *(Note: Please verify this Train/Val/Test year split against your final setup, as previous versions differed slightly).*
* **Sequence Preparation:**
    * Data sorted chronologically. Missing values imputed. Features scaled. Sequences padded/truncated: S2 to 30 steps, SAR to 60 steps.

### 3. Modeling for Early Prediction

As for the improvements to our model, we are now using a Dual-Input Transformer, utilising both the SAR and S2 data, instead of the Vanilla, single-input transformer utilised in the baseline for detection. This approach was chosen because, firstly, the satellites that capture SAR and S2 data do not orbit over a single location at the same time. Furthermore, various factors, such as high cloud cover, may render many S2 samples useless (if cloud cover in a sample was higher than 40%, it was not included). Therefore, if we were to merge the datasets, a lot of SAR samples would have to be discarded too, resulting in further loss of the already very limited data points.

* **Baseline Model (S2 Transformer - Original Detection Task):**
    * An initial Transformer model for *burn detection*.
    * **Notebook:** `deliverable_3.ipynb`.
* **Random Forest Baseline (Prediction Task):**
    * Trained on static statistical features. Performance: 50% Accuracy, 0.00 F1-Score.
* **Dual-Input Transformer (Prediction Task - Main Model):**
    * Architecture: Parallel S2/SAR branches (Embedding -> Positional Encoding -> Transformer Encoder -> Pooling) -> Concatenation -> Dense Layers.
    * Upon implementing the transformer, we realised the model was overfitting quite drastically (test accuracy ~70%). Therefore, we made the model more complex by adding dropout and regularization, the results of which are markedly better.
    * Performance (on its respective test set): ≈99.14% Accuracy, ≈0.9915 F1-Score.
    * To show the effectiveness of our transformer, we have compared its results with the Random Forest model.
    * **Notebook:** `deliverable4.ipynb`.

### 4. Burned Area Segmentation 

To complement the prediction of *likelihood*, this phase aims to estimate the *spatial extent* of burned areas post-event using image segmentation. Image segmentation is a fundamental computer vision technique crucial for environmental monitoring as it allows for precise spatial quantification, detailed mapping of land cover, change detection, and targeted analysis of events like fires.

* **Current Work (Forest Fire Segmentation with Enhanced Inputs):** Our current development focuses on segmenting burned areas from **forest fires** using the CEMS (Copernicus Emergency Management Service) Wildfires dataset (`links-ads/wildfires-cems`). This dataset contains Sentinel-2 images paired with manually delineated masks of burned areas.
* **Model:** We are utilizing a semantic segmentation model, currently configured with the **SegFormer** architecture.
* **Key Improvement - Spectral Indices:** To enhance the model's ability to distinguish burned areas, we incorporate spectral indices as additional input channels alongside raw Sentinel-2 bands. Specifically, the **Normalized Burn Ratio (NBR)** `(NIR - SWIR) / (NIR + SWIR)` and the **Normalized Difference Vegetation Index (NDVI)** `(NIR - Red) / (NIR + Red)` are used. NBR is highly sensitive to fire-induced changes, while NDVI provides crucial context about vegetation health.
* **Goal (Stubble Burning Segmentation):** Our goal is to adapt and apply these image segmentation techniques to specifically detect and map areas affected by **stubble burning**.
* **Methodology: Transfer Learning / Fine-tuning:** Instead of training a model from scratch for stubble burning, we plan to leverage our work on the forest fire dataset using Transfer Learning. The model pre-trained on the CEMS forest fire dataset will have learned valuable features for identifying burned biomass from Sentinel-2 data. We will adapt this model by fine-tuning it on a new dataset specifically containing examples of stubble burning events. This process typically involves initializing with pre-trained weights and retraining on the stubble burning dataset, often with a lower learning rate.
    * **Why Transfer Learning?** It reduces the amount of labeled data needed, is generally faster, and often leads to better performance by leveraging robust features learned on a larger source dataset.
* **Notebook (Forest Fire Segmentation & Transfer Learning Plan):** `final_improvement.ipynb`.



## Acknowledgements
Ashhad Ali (My esteemed project partner)
