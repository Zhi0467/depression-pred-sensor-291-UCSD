# DSC 291 Final Project: Enhance Mental Issue Predictions With HRV Wearable Data
We use the dataset from [Baigutanova, A. et al.](https://www.nature.com/articles/s41597-025-05801-3#Sec11)
**Research Question:** Does continuous daytime sensor data improve the prediction of depression (PHQ9) and anxiety (GAD7) scores compared to a baseline model using only sleep diaries and participant surveys?

---

## Dataset Overview

The dataset combines multimodal data from 49 participants over four weeks, located in the data folder.

-   **Surveys (`survey.csv`):** Participant demographics, lifestyle habits, and biweekly clinical scores (PHQ9, GAD7, ISI).
-   **Sleep Diaries (`sleep_diary.csv`):** Daily self-reported sleep metrics like `sleep_duration`, `sleep_efficiency`, and `sleep_latency`.
-   **Processed Sensor Data (`sensor_hrv_filtered.csv`):** 5-minute daytime aggregates of physiological (HR, HRV), motion (steps), and environmental (light) data.
-   **Raw Sensor Data (`raw_data/`):** High-frequency (10 Hz) time-series signals from the smartwatch sensors.

---

## High-Level Approach

Our analysis is a retrospective comparison of three machine learning models to isolate the predictive value of passive sensor data.

### Core Modules

1.  **Data Ingestion:**
    -   Merge survey, sleep, and processed sensor data tables.
    -   Aggregate daily and 5-minute data to align with biweekly PHQ9/GAD7 scores.

2.  **Feature Engineering:**
    -   Extract static features (demographics, lifestyle).
    -   Engineer aggregate sleep features from diaries (e.g., `mean_sleep_duration`).
    -   Engineer aggregate sensor features from daytime data (e.g., `mean_daytime_rmssd`).

3.  **Comparative Modeling:**
    -   Train and evaluate three models using cross-validation:
        1.  **Baseline Model:** Static + Sleep Features
        2.  **Sensor-Only Model:** Static + Sensor Features
        3.  **Combined Model:** Static + Sleep + Sensor Features


---

## Running the Final Model Notebook

The final analysis is implemented as an interactive [marimo](https://marimo.io) notebook located at `src/notebooks/final_model_notebook.py`.

### Prerequisites

- Python >= 3.10
- Install dependencies using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

### Running the Notebook

To launch the interactive notebook:

```bash
uv run marimo run src/notebooks/final_model_notebook.py
```

To edit the notebook in the marimo editor:

```bash
uv run marimo edit src/notebooks/final_model_notebook.py
```

The notebook provides interactive controls for:
- Exploring the dataset (surveys, sleep diaries, sensor data)
- Configuring UMAP dimensionality reduction parameters
- Adjusting regularization strength for Ridge/Lasso models
- Visualizing predicted vs actual scores
- Viewing patient-level time series trends
