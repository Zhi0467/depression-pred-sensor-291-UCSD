# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "pandas",
#     "numpy",
#     "scikit-learn",
#     "plotly",
#     "umap-learn",
#     "altair==6.0.0",
#     "vegafusion==2.0.3",
#     "pgmpy",
#     "networkx",
# ]
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Depression Prediction from Sensor Data: Final Model Analysis

    ## Project Summary
    This project investigates whether continuous sensor data can improve the prediction of clinical mental health scores compared to baseline models using only self-reported surveys and sleep diaries. We use multimodal data from **49 participants** collected over **4 weeks**, including:

    - **Wearable sensor measurements**: Heart rate variability, motion, light exposure
    - **Daily sleep diaries**: Self-reported sleep metrics
    - **Validated clinical assessments**: PHQ-9 (depression), GAD-7 (anxiety), ISI (insomnia)

    ## Objectives
    Build a retrospective comparison study using machine learning to isolate the predictive value of passive sensor data for mental health assessment.

    ## Team Members
    - Joshua Chuang
    - Joyce Hu
    - Aritra Das
    - Zhi Wang

    ---
    """)
    return


@app.cell
def _():
    # Core imports
    import sys
    import os
    import warnings

    warnings.filterwarnings("ignore")

    # Data science stack
    import numpy as np
    import pandas as pd
    import torch

    # Scikit-learn
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, Lasso, LinearRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline

    # UMAP for dimensionality reduction
    import umap

    # Plotly for interactive visualizations
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # # Set random seeds for reproducibility
    # np.random.seed(42)
    # torch.manual_seed(42)

    print("All imports successful!")
    return (
        Lasso,
        LeaveOneOut,
        LinearRegression,
        Ridge,
        StandardScaler,
        cross_val_predict,
        go,
        make_subplots,
        mean_absolute_error,
        mean_squared_error,
        np,
        os,
        pd,
        r2_score,
        sys,
        umap,
    )


@app.cell
def _(os, sys):
    # Setup paths - find project root
    # This works whether running from notebooks dir or project root
    _current_dir = (
        os.path.dirname(os.path.abspath(__file__))
        if "__file__" in dir()
        else os.getcwd()
    )

    # Navigate to project root
    if "notebooks" in _current_dir:
        PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, "..", ".."))
    else:
        PROJECT_ROOT = _current_dir

    # Add to path for imports
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    # Data paths
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    SURVEY_PATH = os.path.join(DATA_DIR, "survey.csv")
    DIARY_PATH = os.path.join(DATA_DIR, "sleep_diary.csv")
    SENSOR_PATH = os.path.join(DATA_DIR, "sensor_hrv.csv")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_logs")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Target labels (clinical scores)
    TARGET_LABELS = ["PHQ9_F", "GAD7_F", "ISI_F"]
    TARGET_NAMES = {
        "PHQ9_F": "Depression (PHQ-9)",
        "GAD7_F": "Anxiety (GAD-7)",
        "ISI_F": "Insomnia (ISI)",
    }

    # Feature column definitions
    SURVEY_FEATURE_COLS = [
        "sex",
        "age",
        "marriage",
        "smartwatch",
        "regular",
        "exercise",
        "coffee",
        "smoking",
        "drinking",
    ]

    DIARY_FEATURE_COLS = [
        "sleep_duration",
        "in_bed_duration",
        "sleep_latency",
        "sleep_efficiency",
        "waso",
        "wakeup@night",
    ]

    SENSOR_FEATURE_COLS = [
        "missingness_score",
        "HR",
        "ibi",
        "steps",
        "calories",
        "light_avg",
        "sdnn",
        "sdsd",
        "rmssd",
        "pnn20",
        "pnn50",
        "lf",
        "hf",
        "lf/hf",
    ]

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    return (
        DIARY_FEATURE_COLS,
        DIARY_PATH,
        OUTPUT_DIR,
        SENSOR_FEATURE_COLS,
        SENSOR_PATH,
        SURVEY_FEATURE_COLS,
        SURVEY_PATH,
        TARGET_LABELS,
        TARGET_NAMES,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1. Data Loading & Exploration

    We load three data sources:
    1. **Survey Data**: Static demographics and lifestyle factors (49 participants)
    2. **Sleep Diary**: Daily self-reported sleep metrics (1,373 records)
    3. **Sensor HRV Data**: High-resolution 5-minute interval measurements (79,640 intervals)
    """)
    return


@app.cell
def _(DIARY_PATH, SENSOR_PATH, SURVEY_PATH, pd):
    # Load raw data
    df_survey = pd.read_csv(SURVEY_PATH)
    df_diary = pd.read_csv(DIARY_PATH)
    df_sensor = pd.read_csv(SENSOR_PATH)

    # Basic data info
    data_summary = {
        "Dataset": ["Survey", "Sleep Diary", "Sensor HRV"],
        "Records": [len(df_survey), len(df_diary), len(df_sensor)],
        "Features": [df_survey.shape[1], df_diary.shape[1], df_sensor.shape[1]],
        "Unique Users": [
            df_survey["deviceId"].nunique(),
            df_diary["userId"].nunique(),
            df_sensor["deviceId"].nunique(),
        ],
    }
    df_data_summary = pd.DataFrame(data_summary)
    print("Data loaded successfully!")
    df_data_summary
    return df_diary, df_sensor, df_survey


@app.cell
def _(mo):
    # Dataset selector for exploration
    dataset_selector = mo.ui.dropdown(
        options=["Survey", "Sleep Diary", "Sensor HRV"],
        value="Survey",
        label="Select Dataset to Explore",
    )
    dataset_selector
    return (dataset_selector,)


@app.cell
def _(dataset_selector, df_diary, df_sensor, df_survey, mo):
    # Show selected dataset preview
    _dataset_map = {
        "Survey": df_survey,
        "Sleep Diary": df_diary,
        "Sensor HRV": df_sensor,
    }
    _selected_df = _dataset_map[dataset_selector.value]

    mo.vstack(
        [
            mo.md(f"### Preview: {dataset_selector.value} Data"),
            mo.md(
                f"**Shape**: {_selected_df.shape[0]} rows x {_selected_df.shape[1]} columns"
            ),
            mo.ui.table(_selected_df, selection=None),
        ]
    )
    return


@app.cell
def _(TARGET_LABELS, TARGET_NAMES, df_survey, go, make_subplots):
    # Create score distribution plots
    fig_distributions = make_subplots(
        rows=1, cols=3, subplot_titles=[TARGET_NAMES[t] for t in TARGET_LABELS]
    )

    _colors = ["#636EFA", "#EF553B", "#00CC96"]

    for _i, _target in enumerate(TARGET_LABELS):
        fig_distributions.add_trace(
            go.Histogram(
                x=df_survey[_target],
                name=TARGET_NAMES[_target],
                marker_color=_colors[_i],
                opacity=0.75,
                nbinsx=15,
            ),
            row=1,
            col=_i + 1,
        )

    fig_distributions.update_layout(
        title_text="Distribution of Clinical Assessment Scores",
        showlegend=False,
        height=400,
    )

    fig_distributions.update_xaxes(title_text="Score")
    fig_distributions.update_yaxes(title_text="Count")
    fig_distributions
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.1 Data Quality Diagnostics

    Before modeling, we identify data quality issues that may impact performance.
    """)
    return


@app.cell
def _(
    DIARY_FEATURE_COLS,
    SENSOR_FEATURE_COLS,
    SURVEY_FEATURE_COLS,
    df_diary,
    df_sensor,
    df_survey,
    go,
    pd,
):
    # Compute missingness for each dataset
    survey_missing = (
        df_survey[SURVEY_FEATURE_COLS].isnull().sum() / len(df_survey) * 100
    )
    diary_missing = df_diary[DIARY_FEATURE_COLS].isnull().sum() / len(df_diary) * 100
    sensor_missing = (
        df_sensor[SENSOR_FEATURE_COLS].isnull().sum() / len(df_sensor) * 100
    )

    # Combine into single dataframe
    missing_data = []
    for _col, _val in survey_missing.items():
        missing_data.append({"Feature": _col, "Source": "Survey", "Missing %": _val})
    for _col, _val in diary_missing.items():
        missing_data.append({"Feature": _col, "Source": "Diary", "Missing %": _val})
    for _col, _val in sensor_missing.items():
        missing_data.append({"Feature": _col, "Source": "Sensor", "Missing %": _val})

    missing_df = pd.DataFrame(missing_data)

    # Create bar chart of missingness (only features with >0% missing)
    missing_df_filtered = missing_df[missing_df["Missing %"] > 0].sort_values(
        "Missing %", ascending=True
    )

    fig_missing = go.Figure()

    color_map = {"Survey": "#636EFA", "Diary": "#EF553B", "Sensor": "#00CC96"}

    for source in ["Survey", "Diary", "Sensor"]:
        source_data = missing_df_filtered[missing_df_filtered["Source"] == source]
        if len(source_data) > 0:
            fig_missing.add_trace(
                go.Bar(
                    y=source_data["Feature"],
                    x=source_data["Missing %"],
                    orientation="h",
                    name=source,
                    marker_color=color_map[source],
                )
            )

    fig_missing.update_layout(
        title="Features with Missing Data",
        xaxis_title="Missing %",
        yaxis_title="Feature",
        height=400,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig_missing
    return


@app.cell
def _(SENSOR_FEATURE_COLS, df_sensor, mo, pd):
    # Detect corrupted features (extreme values) and compute quality metrics
    feature_quality = []

    for _col in SENSOR_FEATURE_COLS:
        _max_val = df_sensor[_col].max()
        _min_val = df_sensor[_col].min()
        _missing_pct = df_sensor[_col].isnull().sum() / len(df_sensor) * 100

        _issues = []
        if _max_val > 1e10:
            _issues.append(f"Extreme max: {_max_val:.2e}")
        if _missing_pct > 20:
            _issues.append(f"High missing: {_missing_pct:.1f}%")

        feature_quality.append(
            {
                "Feature": _col,
                "Min": f"{_min_val:.2f}" if abs(_min_val) < 1e6 else f"{_min_val:.2e}",
                "Max": f"{_max_val:.2f}" if abs(_max_val) < 1e6 else f"{_max_val:.2e}",
                "Missing %": f"{_missing_pct:.1f}%",
                "Issues": ", ".join(_issues) if _issues else "OK",
            }
        )

    feature_quality_df = pd.DataFrame(feature_quality)
    problematic_features = feature_quality_df[feature_quality_df["Issues"] != "OK"]

    mo.vstack(
        [
            mo.callout(
                mo.md(f"""
    **Data Quality Issues Found ({len(problematic_features)} sensor features with issues):**

    - `lf`, `hf` have values up to **1e+46**
    - `steps`, `calories`, `distance` have **~55% missing** values

    These issues are highlighted in the Feature Selection section below.
            """),
                kind="warn",
            ),
            mo.md("**Sensor Feature Quality Summary:**"),
            mo.ui.table(feature_quality_df, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1.2 Target Variable Analysis

    Understanding the distribution of clinical scores and class imbalance.
    """)
    return


@app.cell
def _(TARGET_LABELS, df_survey, go, make_subplots):
    # Clinical severity cutoffs
    SEVERITY_CUTOFFS = {
        "PHQ9_F": [
            (0, 4, "Minimal"),
            (5, 9, "Mild"),
            (10, 14, "Moderate"),
            (15, 19, "Mod-Severe"),
            (20, 27, "Severe"),
        ],
        "GAD7_F": [
            (0, 4, "Minimal"),
            (5, 9, "Mild"),
            (10, 14, "Moderate"),
            (15, 21, "Severe"),
        ],
        "ISI_F": [
            (0, 7, "None"),
            (8, 14, "Subthreshold"),
            (15, 21, "Moderate"),
            (22, 28, "Severe"),
        ],
    }

    # Count participants in each severity category
    severity_counts = {}
    for _target in TARGET_LABELS:
        _counts = []
        for _low, _high, _label in SEVERITY_CUTOFFS[_target]:
            _count = (
                (df_survey[_target] >= _low) & (df_survey[_target] <= _high)
            ).sum()
            _counts.append(
                {
                    "Severity": _label,
                    "Count": _count,
                    "Percentage": _count / len(df_survey) * 100,
                }
            )
        severity_counts[_target] = _counts

    # Create pie charts for each target
    fig_severity = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["PHQ-9 (Depression)", "GAD-7 (Anxiety)", "ISI (Insomnia)"],
        specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]],
    )

    _colors_severity = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#9b59b6"]

    for _idx, _target in enumerate(TARGET_LABELS):
        _labels = [c["Severity"] for c in severity_counts[_target]]
        _values = [c["Count"] for c in severity_counts[_target]]

        fig_severity.add_trace(
            go.Pie(
                labels=_labels,
                values=_values,
                marker_colors=_colors_severity[: len(_labels)],
                textinfo="label+percent",
                textposition="inside",
                hole=0.3,
                showlegend=(_idx == 0),
            ),
            row=1,
            col=_idx + 1,
        )

    fig_severity.update_layout(
        title_text="Clinical Severity Distribution (n=49)", height=400
    )
    fig_severity
    return


@app.cell
def _(df_survey, mo):
    # Compute class distribution and show warning
    phq9_minimal = (df_survey["PHQ9_F"] <= 4).sum()
    phq9_mild = ((df_survey["PHQ9_F"] >= 5) & (df_survey["PHQ9_F"] <= 9)).sum()
    phq9_clinical = (df_survey["PHQ9_F"] >= 10).sum()

    gad7_minimal = (df_survey["GAD7_F"] <= 4).sum()
    gad7_clinical = (df_survey["GAD7_F"] >= 10).sum()

    isi_none = (df_survey["ISI_F"] <= 7).sum()
    isi_clinical = (df_survey["ISI_F"] >= 15).sum()

    n_participants = len(df_survey)

    mo.callout(
        mo.md(f"""
    **Class Imbalance Warning:**

    | Target | Minimal/None | Mild/Subthreshold | Clinical (Moderate+) |
    |--------|--------------|-------------------|----------------------|
    | PHQ-9  | {phq9_minimal} ({phq9_minimal / n_participants * 100:.0f}%) | {phq9_mild} ({phq9_mild / n_participants * 100:.0f}%) | **{phq9_clinical} ({phq9_clinical / n_participants * 100:.0f}%)** |
    | GAD-7  | {gad7_minimal} ({gad7_minimal / n_participants * 100:.0f}%) | {n_participants - gad7_minimal - gad7_clinical} ({(n_participants - gad7_minimal - gad7_clinical) / n_participants * 100:.0f}%) | **{gad7_clinical} ({gad7_clinical / n_participants * 100:.0f}%)** |
    | ISI    | {isi_none} ({isi_none / n_participants * 100:.0f}%) | {n_participants - isi_none - isi_clinical} ({(n_participants - isi_none - isi_clinical) / n_participants * 100:.0f}%) | **{isi_clinical} ({isi_clinical / n_participants * 100:.0f}%)** |

    **Impact:** The vast majority of participants have minimal/no symptoms. 
    With so few clinical cases, the model might lacks examples to learn from and tends to predict "average" (low) scores for everyone.
        """),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2. Feature Engineering

    ### Approach:
    1. **Aggregate temporal sequences** (diary, sensor) to fixed-length vectors using statistical summaries (mean, std)
    2. **Apply StandardScaler** to normalize features after aggregation
    3. **Use UMAP** for dimensionality reduction on high-dimensional sensor features

    This addresses the overfitting issue observed in baseline models where high-dimensional sensor features (56 after aggregation) caused severe degradation with only 49 samples.
    """)
    return


@app.cell
def _(np):
    import numpy

    def aggregate_sequences(seq_data: dict, feature_cols: list) -> numpy.ndarray:
        """
        Aggregate variable-length sequences to fixed-length features.
        Uses only mean and std (dropped min/max to reduce dimensionality).

        Args:
            seq_data: Dictionary mapping user_id -> DataFrame of sequence data
            feature_cols: List of feature column names

        Returns:
            Aggregated features array of shape (n_users, n_features * 2)
        """
        aggregated = []

        for user_id in sorted(seq_data.keys()):
            df = seq_data[user_id]
            # Fill NaN values before aggregation
            features = df[feature_cols].fillna(df[feature_cols].mean()).values

            if len(features) == 0:
                # Handle empty sequences
                stats = np.zeros(len(feature_cols) * 2)
            else:
                # Use nanmean/nanstd to handle any remaining NaN values
                mean_vals = np.nanmean(features, axis=0)
                std_vals = np.nanstd(features, axis=0)
                # Replace any NaN results with 0 (e.g., from empty slices)
                mean_vals = np.nan_to_num(mean_vals, nan=0.0)
                std_vals = np.nan_to_num(std_vals, nan=0.0)
                stats = np.concatenate([mean_vals, std_vals])

            aggregated.append(stats)

        return np.array(aggregated)

    def prepare_sequence_data(df, id_col, feature_cols):
        """Group dataframe by user ID and return dictionary."""
        seq_data = {}
        for user_id, group in df.groupby(id_col):
            seq_data[user_id] = group[feature_cols].copy()
        return seq_data
    return aggregate_sequences, prepare_sequence_data


@app.cell
def _(TARGET_LABELS, df_diary, df_sensor, df_survey):
    # Get intersection of users across all datasets (computed once)
    survey_users = set(df_survey["deviceId"].unique())
    diary_users = set(df_diary["userId"].unique())
    sensor_users = set(df_sensor["deviceId"].unique())

    common_users = sorted(survey_users & diary_users & sensor_users)
    n_users = len(common_users)
    print(f"Common users across all datasets: {n_users}")

    # Prepare base survey data and labels (these don't change with feature selection)
    survey_data = df_survey[df_survey["deviceId"].isin(common_users)].sort_values(
        "deviceId"
    )
    y_all = survey_data[TARGET_LABELS].fillna(0).values
    user_ids = survey_data["deviceId"].tolist()
    return user_ids, y_all


@app.cell
def _(mo):
    mo.md("""
    ### 2.1 Feature Selection

    Select which features to include in model training. Features with data quality issues are marked with `[!]`.

    **Instructions:**
    1. Check/uncheck features you want to include
    2. Use the buttons to quickly select/clear groups
    3. Click **"Apply Feature Selection"** to confirm and retrain the model
    """)
    return


@app.cell
def _(DIARY_FEATURE_COLS, SENSOR_FEATURE_COLS, SURVEY_FEATURE_COLS, mo):
    # Define problematic features (corrupted or high missing)
    PROBLEMATIC_SENSOR_FEATURES = {"lf", "hf", "steps", "calories"}

    # Create feature multiselect for Survey
    survey_multiselect = mo.ui.multiselect(
        options=SURVEY_FEATURE_COLS,
        value=list(SURVEY_FEATURE_COLS),  # All selected by default
        label="Survey Features",
    )

    # Create feature multiselect for Diary
    diary_multiselect = mo.ui.multiselect(
        options=DIARY_FEATURE_COLS,
        value=list(DIARY_FEATURE_COLS),  # All selected by default
        label="Diary Features",
    )

    # Create feature multiselect for Sensor (problematic ones unselected by default)
    _sensor_default = [
        c for c in SENSOR_FEATURE_COLS if c not in PROBLEMATIC_SENSOR_FEATURES
    ]
    _sensor_options = {
        (f"{c} [!]" if c in PROBLEMATIC_SENSOR_FEATURES else c): c
        for c in SENSOR_FEATURE_COLS
    }
    sensor_multiselect = mo.ui.multiselect(
        options=_sensor_options,
        value=_sensor_default,  # Problematic features unselected by default
        label="Sensor Features",
    )
    return diary_multiselect, sensor_multiselect, survey_multiselect


@app.cell
def _(mo, survey_multiselect):
    # Survey features section with Select All / Clear All
    mo.vstack(
        [
            mo.md("**Survey Features (Demographics & Lifestyle):**"),
            survey_multiselect,
        ]
    )
    return


@app.cell
def _(diary_multiselect, mo):
    # Diary features section
    mo.vstack(
        [
            mo.md("**Sleep Diary Features:**"),
            diary_multiselect,
        ]
    )
    return


@app.cell
def _(mo, sensor_multiselect):
    # Sensor features section
    mo.vstack(
        [
            mo.md("**Sensor Features (HRV & Activity):** _[!] = data quality issues_"),
            sensor_multiselect,
        ]
    )
    return


@app.cell
def _(mo):
    # Action buttons for feature selection
    apply_selection_btn = mo.ui.button(label="Apply Feature Selection", kind="success")
    return (apply_selection_btn,)


@app.cell
def _(apply_selection_btn, mo):
    # Display the apply button
    mo.hstack(
        [apply_selection_btn, mo.md("_Click to apply changes and retrain model_")],
        justify="start",
        gap=2,
    )
    return


@app.cell
def _(
    DIARY_FEATURE_COLS,
    SENSOR_FEATURE_COLS,
    SURVEY_FEATURE_COLS,
    apply_selection_btn,
    diary_multiselect,
    mo,
    sensor_multiselect,
    survey_multiselect,
):
    # When button is clicked, capture the current selection as "confirmed"
    # The button click triggers this cell to re-run
    _click_count = apply_selection_btn.value

    # Get selected features from multiselects
    selected_survey_features = list(survey_multiselect.value)
    selected_diary_features = list(diary_multiselect.value)
    selected_sensor_features = list(sensor_multiselect.value)

    # Calculate totals
    _total_selected = (
        len(selected_survey_features)
        + len(selected_diary_features)
        + len(selected_sensor_features)
    )
    _total_available = (
        len(SURVEY_FEATURE_COLS) + len(DIARY_FEATURE_COLS) + len(SENSOR_FEATURE_COLS)
    )

    # Calculate aggregated feature count (mean + std for diary and sensor)
    total_model_features = (
        len(selected_survey_features)
        + len(selected_diary_features) * 2  # mean + std
        + len(selected_sensor_features) * 2  # mean + std
    )

    # Determine if ratio is good or bad
    _ratio = total_model_features / 49 if total_model_features > 0 else 0
    _ratio_status = "good" if _ratio < 0.5 else ("warn" if _ratio < 1.0 else "danger")

    mo.vstack(
        [
            mo.callout(
                mo.md(f"""
    **Current Feature Selection (Applied):**
    - Survey: **{len(selected_survey_features)}/{len(SURVEY_FEATURE_COLS)}** features
    - Diary: **{len(selected_diary_features)}/{len(DIARY_FEATURE_COLS)}** features ({len(selected_diary_features) * 2} after aggregation)
    - Sensor: **{len(selected_sensor_features)}/{len(SENSOR_FEATURE_COLS)}** features ({len(selected_sensor_features) * 2} after aggregation)
    - **Total model features: {total_model_features}** (with n=49 samples, ratio = {_ratio:.2f} features/sample)
            """),
                kind="success"
                if _ratio_status == "good"
                else ("warn" if _ratio_status == "warn" else "danger"),
            ),
            mo.md(
                f"_Recommendation: Keep ratio below 0.5 for best results (currently {_ratio:.2f})_"
            )
            if _ratio >= 0.5
            else mo.md(""),
        ]
    )
    return (
        selected_diary_features,
        selected_sensor_features,
        selected_survey_features,
    )


@app.cell
def _(
    aggregate_sequences,
    df_diary,
    df_sensor,
    df_survey,
    np,
    prepare_sequence_data,
    selected_diary_features,
    selected_sensor_features,
    selected_survey_features,
):
    # Prepare data using SELECTED features
    # Get intersection of users across all datasets
    _survey_users = set(df_survey["deviceId"].unique())
    _diary_users = set(df_diary["userId"].unique())
    _sensor_users = set(df_sensor["deviceId"].unique())
    _common_users = sorted(_survey_users & _diary_users & _sensor_users)

    # Prepare survey features with SELECTED columns
    _survey_data = df_survey[df_survey["deviceId"].isin(_common_users)].sort_values(
        "deviceId"
    )

    if len(selected_survey_features) > 0:
        X_survey_raw = (
            _survey_data[selected_survey_features]
            .fillna(_survey_data[selected_survey_features].mean())
            .values
        )
    else:
        X_survey_raw = np.zeros((len(_common_users), 0))

    # Prepare diary sequence data with SELECTED columns
    if len(selected_diary_features) > 0:
        diary_seq = prepare_sequence_data(
            df_diary[df_diary["userId"].isin(_common_users)],
            "userId",
            selected_diary_features,
        )
        X_diary_raw = aggregate_sequences(diary_seq, selected_diary_features)
    else:
        X_diary_raw = np.zeros((len(_common_users), 0))

    # Prepare sensor sequence data with SELECTED columns
    if len(selected_sensor_features) > 0:
        sensor_seq = prepare_sequence_data(
            df_sensor[df_sensor["deviceId"].isin(_common_users)],
            "deviceId",
            selected_sensor_features,
        )
        X_sensor_raw = aggregate_sequences(sensor_seq, selected_sensor_features)
    else:
        X_sensor_raw = np.zeros((len(_common_users), 0))

    # Print shapes
    print(f"Feature dimensions (using selected features):")
    print(f"  Survey: {X_survey_raw.shape}")
    print(f"  Diary (aggregated): {X_diary_raw.shape}")
    print(f"  Sensor (aggregated): {X_sensor_raw.shape}")
    return X_diary_raw, X_sensor_raw, X_survey_raw


@app.cell
def _(mo):
    # UMAP configuration controls
    mo.md("""
    ### UMAP Dimensionality Reduction Settings
    """)
    return


@app.cell
def _(mo):
    umap_enabled = mo.ui.checkbox(value=True, label="Enable UMAP for sensor features")
    umap_n_components = mo.ui.slider(
        start=2, stop=15, value=5, step=1, label="UMAP Components"
    )
    umap_n_neighbors = mo.ui.slider(
        start=5, stop=30, value=15, step=1, label="UMAP Neighbors"
    )
    umap_min_dist = mo.ui.slider(
        start=0.0, stop=1.0, value=0.1, step=0.05, label="UMAP Min Distance"
    )

    mo.hstack(
        [umap_enabled, umap_n_components, umap_n_neighbors, umap_min_dist],
        justify="start",
        gap=2,
    )
    return umap_enabled, umap_min_dist, umap_n_components, umap_n_neighbors


@app.cell
def _(
    StandardScaler,
    X_diary_raw,
    X_sensor_raw,
    X_survey_raw,
    np,
    umap,
    umap_enabled,
    umap_min_dist,
    umap_n_components,
    umap_n_neighbors,
):
    # Apply StandardScaler to all features (handle empty arrays)
    scaler_survey = StandardScaler()
    scaler_diary = StandardScaler()
    scaler_sensor = StandardScaler()

    # Scale features if they exist, otherwise keep empty array
    if X_survey_raw.shape[1] > 0:
        X_survey_scaled = scaler_survey.fit_transform(X_survey_raw)
        X_survey_scaled = np.nan_to_num(
            X_survey_scaled, nan=0.0, posinf=0.0, neginf=0.0
        )
    else:
        X_survey_scaled = X_survey_raw

    if X_diary_raw.shape[1] > 0:
        X_diary_scaled = scaler_diary.fit_transform(X_diary_raw)
        X_diary_scaled = np.nan_to_num(X_diary_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        X_diary_scaled = X_diary_raw

    if X_sensor_raw.shape[1] > 0:
        X_sensor_scaled = scaler_sensor.fit_transform(X_sensor_raw)
        X_sensor_scaled = np.nan_to_num(
            X_sensor_scaled, nan=0.0, posinf=0.0, neginf=0.0
        )
    else:
        X_sensor_scaled = X_sensor_raw

    # Apply UMAP to sensor features if enabled and there are enough features
    if umap_enabled.value and X_sensor_scaled.shape[1] >= 2:
        n_components = min(umap_n_components.value, X_sensor_scaled.shape[1])
        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(umap_n_neighbors.value, X_sensor_scaled.shape[0] - 1),
            min_dist=umap_min_dist.value,
            random_state=42,
        )
        X_sensor_reduced = umap_model.fit_transform(X_sensor_scaled)
        sensor_label = f"Sensor (UMAP {n_components}D)"
    else:
        X_sensor_reduced = X_sensor_scaled
        sensor_label = (
            "Sensor (scaled)"
            if X_sensor_scaled.shape[1] > 0
            else "Sensor (none selected)"
        )

    # Create feature matrices for each configuration
    # Config 1: Survey + Diary
    X_survey_diary = np.hstack([X_survey_scaled, X_diary_scaled])

    # Config 2: Survey + Sensor
    X_survey_sensor = np.hstack([X_survey_scaled, X_sensor_reduced])

    # Config 3: Survey + Diary + Sensor
    X_all = np.hstack([X_survey_scaled, X_diary_scaled, X_sensor_reduced])

    print(f"Feature dimensions after processing:")
    print(f"  Survey (scaled): {X_survey_scaled.shape}")
    print(f"  Diary (scaled): {X_diary_scaled.shape}")
    print(f"  {sensor_label}: {X_sensor_reduced.shape}")
    print(f"\nConfiguration dimensions:")
    print(f"  Survey + Diary: {X_survey_diary.shape}")
    print(f"  Survey + Sensor: {X_survey_sensor.shape}")
    print(f"  Survey + Diary + Sensor: {X_all.shape}")
    return (
        X_all,
        X_diary_scaled,
        X_sensor_reduced,
        X_sensor_scaled,
        X_survey_diary,
        X_survey_scaled,
        X_survey_sensor,
        sensor_label,
    )


@app.cell
def _(
    TARGET_LABELS,
    TARGET_NAMES,
    X_sensor_reduced,
    go,
    make_subplots,
    umap_enabled,
    umap_n_components,
    y_all,
):
    # UMAP visualization (2D projection for visualization)
    fig_umap = None
    if umap_enabled.value and umap_n_components.value >= 2:
        # Use first 2 components for visualization
        fig_umap = make_subplots(
            rows=1, cols=3, subplot_titles=[TARGET_NAMES[t] for t in TARGET_LABELS]
        )

        for _idx, _target in enumerate(TARGET_LABELS):
            fig_umap.add_trace(
                go.Scatter(
                    x=X_sensor_reduced[:, 0],
                    y=X_sensor_reduced[:, 1],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=y_all[:, _idx],
                        colorscale="Viridis",
                        showscale=(_idx == 2),
                        colorbar=dict(title="Score") if _idx == 2 else None,
                    ),
                    name=TARGET_NAMES[_target],
                    hovertemplate=f"UMAP1: %{{x:.2f}}<br>UMAP2: %{{y:.2f}}<br>{_target}: %{{marker.color}}<extra></extra>",
                ),
                row=1,
                col=_idx + 1,
            )

        fig_umap.update_layout(
            title_text="UMAP Projection of Sensor Features (colored by clinical scores)",
            height=400,
            showlegend=False,
        )
        fig_umap.update_xaxes(title_text="UMAP 1")
        fig_umap.update_yaxes(title_text="UMAP 2")
    else:
        print(
            "UMAP visualization requires at least 2 components. Enable UMAP and set components >= 2."
        )
    fig_umap
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.2 Feature-Target Correlations

    Understanding which features correlate with clinical outcomes helps explain model behavior.
    """)
    return


@app.cell
def _(
    TARGET_LABELS,
    X_all,
    X_sensor_reduced,
    go,
    np,
    pd,
    selected_diary_features,
    selected_sensor_features,
    selected_survey_features,
    umap_enabled,
    y_all,
):
    # Build feature names for the combined feature matrix
    # Note: When UMAP is enabled, sensor features are reduced to UMAP components
    if (
        umap_enabled.value
        and X_sensor_reduced.shape[1] < len(selected_sensor_features) * 2
    ):
        # UMAP was applied - use generic component names
        _sensor_feature_names = [f"UMAP_{i}" for i in range(X_sensor_reduced.shape[1])]
    else:
        # No UMAP or not enough features - use original names
        _sensor_feature_names = [f"{f}_mean" for f in selected_sensor_features] + [
            f"{f}_std" for f in selected_sensor_features
        ]

    _feature_names = (
        list(selected_survey_features)
        + [f"{f}_mean" for f in selected_diary_features]
        + [f"{f}_std" for f in selected_diary_features]
        + _sensor_feature_names
    )

    # Compute correlations if we have features
    if X_all.shape[1] > 0 and len(_feature_names) > 0:
        # Create correlation matrix between features and targets
        _corr_data = []
        for i, feat_name in enumerate(_feature_names):
            for j, target in enumerate(TARGET_LABELS):
                # Compute Pearson correlation
                corr = np.corrcoef(X_all[:, i], y_all[:, j])[0, 1]
                _corr_data.append(
                    {
                        "Feature": feat_name,
                        "Target": target,
                        "Correlation": corr if not np.isnan(corr) else 0,
                    }
                )

        corr_df = pd.DataFrame(_corr_data)
        corr_pivot = corr_df.pivot(
            index="Feature", columns="Target", values="Correlation"
        )

        # Sort by absolute correlation with PHQ9
        corr_pivot["abs_phq9"] = corr_pivot["PHQ9_F"].abs()
        corr_pivot = corr_pivot.sort_values("abs_phq9", ascending=True)
        corr_pivot = corr_pivot.drop("abs_phq9", axis=1)

        # Create heatmap
        fig_correlation = go.Figure(
            data=go.Heatmap(
                z=corr_pivot.values,
                x=TARGET_LABELS,
                y=corr_pivot.index.tolist(),
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr_pivot.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Correlation"),
            )
        )

        fig_correlation.update_layout(
            title="Feature-Target Correlation Heatmap",
            xaxis_title="Target Variable",
            yaxis_title="Feature",
            height=max(400, len(_feature_names) * 20),
        )
    else:
        fig_correlation = None
        corr_pivot = None
        print("No features selected - cannot compute correlations")

    fig_correlation
    return (corr_pivot,)


@app.cell
def _(TARGET_LABELS, corr_pivot, mo, pd):
    # Show top correlated features for each target
    if corr_pivot is not None and len(corr_pivot) > 0:
        _top_features_data = []
        for _target in TARGET_LABELS:
            # Top 3 positive correlations
            _top_pos = corr_pivot[_target].nlargest(3)
            # Top 3 negative correlations
            _top_neg = corr_pivot[_target].nsmallest(3)

            for _feat, _corr in _top_pos.items():
                _top_features_data.append(
                    {
                        "Target": _target,
                        "Feature": _feat,
                        "Correlation": f"{_corr:.3f}",
                        "Direction": "Positive",
                    }
                )
            for _feat, _corr in _top_neg.items():
                _top_features_data.append(
                    {
                        "Target": _target,
                        "Feature": _feat,
                        "Correlation": f"{_corr:.3f}",
                        "Direction": "Negative",
                    }
                )

        _top_features_df = pd.DataFrame(_top_features_data)

        mo.vstack(
            [
                mo.md("**Top Correlated Features per Target:**"),
                mo.ui.table(_top_features_df, selection=None),
            ]
        )
    else:
        mo.md("_No correlation data available_")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3. Baseline Models (Scikit-learn)

    We use **Ridge** (L2) and **Lasso** (L1) regression with **Leave-One-Out Cross-Validation** across all 49 participants.

    ### Model Configurations:
    1. **Survey + Diary**: Demographics + sleep diary features
    2. **Survey + Sensor**: Demographics + sensor features (with UMAP reduction)
    3. **Survey + Diary + Sensor**: All features combined
    """)
    return


@app.cell
def _(mo):
    # Model configuration controls
    alpha_slider = mo.ui.slider(
        start=-4, stop=2, value=0, step=0.5, label="Regularization (log10 alpha)"
    )
    return (alpha_slider,)


@app.cell
def _(alpha_slider, mo):
    mo.md(f"""
    **Regularization Strength**: alpha = 10^{alpha_slider.value} = {10**alpha_slider.value:.4f}
    """)
    return


@app.cell
def _(alpha_slider):
    alpha_slider
    return


@app.cell
def _(
    Lasso,
    LeaveOneOut,
    Ridge,
    TARGET_LABELS,
    X_all,
    X_survey_diary,
    X_survey_sensor,
    alpha_slider,
    cross_val_predict,
    mean_absolute_error,
    mean_squared_error,
    np,
    pd,
    r2_score,
    y_all,
):
    # Run experiments with current alpha
    alpha = 10**alpha_slider.value

    # Configuration mapping
    CONFIGS = {
        "Survey + Diary": X_survey_diary,
        "Survey + Sensor": X_survey_sensor,
        "Survey + Diary + Sensor": X_all,
    }

    REG_TYPES = {"Ridge": Ridge, "Lasso": Lasso}

    results_list = []
    predictions_store = {}

    for _config_name, _X in CONFIGS.items():
        for _reg_name, _RegClass in REG_TYPES.items():
            for _target_idx, _target_name in enumerate(TARGET_LABELS):
                _y = y_all[:, _target_idx]

                # Create model with current alpha
                # For Lasso, use larger max_iter for convergence
                if _reg_name == "Lasso":
                    _model = _RegClass(alpha=alpha, max_iter=10000)
                else:
                    _model = _RegClass(alpha=alpha)

                # Leave-One-Out Cross-Validation
                _loo = LeaveOneOut()
                _predictions = cross_val_predict(_model, _X, _y, cv=_loo)

                # Compute metrics
                _mae = mean_absolute_error(_y, _predictions)
                _rmse = np.sqrt(mean_squared_error(_y, _predictions))
                _r2 = r2_score(_y, _predictions)

                # Store results
                results_list.append(
                    {
                        "Configuration": _config_name,
                        "Regularization": _reg_name,
                        "Target": _target_name,
                        "Alpha": alpha,
                        "MAE": _mae,
                        "RMSE": _rmse,
                        "R2": _r2,
                        "Features": _X.shape[1],
                    }
                )

                # Store predictions for visualization
                _key = (_config_name, _reg_name, _target_name)
                predictions_store[_key] = {"predictions": _predictions, "actuals": _y}

    results_df = pd.DataFrame(results_list)
    print(f"Experiments completed with alpha = {alpha:.4f}")
    return alpha, predictions_store, results_df


@app.cell
def _(mo, results_df):
    # Display results table
    mo.vstack(
        [mo.md("### Results Summary"), mo.ui.table(results_df.round(4), selection=None)]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3.1 Feature Importance Analysis

    Which features have the strongest influence on predictions? Select a configuration and target to see coefficient magnitudes.
    """)
    return


@app.cell
def _(TARGET_LABELS, mo):
    # Selector for feature importance visualization
    importance_target_selector = mo.ui.dropdown(
        options=TARGET_LABELS, value=TARGET_LABELS[0], label="Target Variable"
    )
    importance_config_selector = mo.ui.dropdown(
        options=["Survey + Diary", "Survey + Sensor", "Survey + Diary + Sensor"],
        value="Survey + Diary + Sensor",
        label="Configuration",
    )
    mo.hstack([importance_config_selector, importance_target_selector], gap=2)
    return importance_config_selector, importance_target_selector


@app.cell
def _(
    Ridge,
    TARGET_LABELS,
    X_all,
    X_survey_diary,
    X_survey_sensor,
    alpha,
    go,
    importance_config_selector,
    importance_target_selector,
    np,
    pd,
    selected_diary_features,
    selected_sensor_features,
    selected_survey_features,
    y_all,
):
    # Train Ridge model on full data to get coefficients for feature importance
    _config_map = {
        "Survey + Diary": X_survey_diary,
        "Survey + Sensor": X_survey_sensor,
        "Survey + Diary + Sensor": X_all,
    }

    _X = _config_map[importance_config_selector.value]
    _target_idx = TARGET_LABELS.index(importance_target_selector.value)
    _y = y_all[:, _target_idx]

    # Build feature names based on configuration
    if importance_config_selector.value == "Survey + Diary":
        _feature_names = (
            list(selected_survey_features)
            + [f"{f}_mean" for f in selected_diary_features]
            + [f"{f}_std" for f in selected_diary_features]
        )
    elif importance_config_selector.value == "Survey + Sensor":
        _feature_names = (
            list(selected_survey_features)
            + [f"{f}_mean" for f in selected_sensor_features]
            + [f"{f}_std" for f in selected_sensor_features]
        )
    else:  # Survey + Diary + Sensor
        _feature_names = (
            list(selected_survey_features)
            + [f"{f}_mean" for f in selected_diary_features]
            + [f"{f}_std" for f in selected_diary_features]
            + [f"{f}_mean" for f in selected_sensor_features]
            + [f"{f}_std" for f in selected_sensor_features]
        )

    # Train model and get coefficients
    if _X.shape[1] > 0 and len(_feature_names) == _X.shape[1]:
        _model = Ridge(alpha=alpha)
        _model.fit(_X, _y)

        # Create coefficient dataframe
        coef_df = pd.DataFrame(
            {
                "Feature": _feature_names,
                "Coefficient": _model.coef_,
                "Abs_Coefficient": np.abs(_model.coef_),
            }
        ).sort_values("Abs_Coefficient", ascending=True)

        # Take top 15 features for visualization
        _top_n = min(15, len(coef_df))
        _plot_df = coef_df.tail(_top_n)

        # Create horizontal bar chart
        fig_importance = go.Figure()

        # Color bars based on positive/negative
        _colors = ["#EF553B" if c < 0 else "#636EFA" for c in _plot_df["Coefficient"]]

        fig_importance.add_trace(
            go.Bar(
                y=_plot_df["Feature"],
                x=_plot_df["Coefficient"],
                orientation="h",
                marker_color=_colors,
                text=_plot_df["Coefficient"].round(3),
                textposition="outside",
            )
        )

        fig_importance.update_layout(
            title=f"Top {_top_n} Feature Coefficients<br><sub>{importance_config_selector.value} | {importance_target_selector.value} | Ridge (alpha={alpha:.4f})</sub>",
            xaxis_title="Coefficient Value",
            yaxis_title="Feature",
            height=max(400, _top_n * 25),
            showlegend=False,
        )
        fig_importance.add_vline(x=0, line_dash="dash", line_color="gray")
    else:
        fig_importance = None
        coef_df = None
        print("Cannot compute feature importance - no features or dimension mismatch")

    fig_importance
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4. Bayesian Causal Analysis

    In this section, we use **Greedy Equivalence Search (GES)** to learn the causal structure among features and mental health outcomes. Unlike correlation analysis which only shows associations, Bayesian Belief Networks (BBNs) model directed dependencies that suggest potential causal relationships.

    ### Approach:
    1. Combine selected features (survey, diary, sensor) with target variables (PHQ-9, GAD-7, ISI)
    2. Learn DAG structure using GES algorithm with BIC scoring
    3. Visualize the learned causal graph
    4. Evaluate predictive performance using Leave-One-Out CV
    """)
    return


@app.cell
def _():
    # Import pgmpy for Bayesian Network structure learning
    from pgmpy.estimators import GES
    from pgmpy.estimators import BICGauss
    import networkx as nx

    print("pgmpy imports successful!")
    return BICGauss, GES, nx


@app.cell
def _(mo):
    mo.callout(
        mo.md("""
    **Sample Size Warning:**

    With only **n=49 participants** and potentially many features, structure learning algorithms may:
    - Learn sparse or empty DAGs (few/no edges detected)
    - Miss true causal relationships due to insufficient statistical power
    - Be sensitive to the specific feature selection

    For more robust causal discovery, consider reducing the number of features or collecting more samples.
        """),
        kind="warn",
    )
    return


@app.cell
def _(
    TARGET_LABELS,
    X_diary_scaled,
    X_sensor_scaled,
    X_survey_scaled,
    np,
    pd,
    selected_diary_features,
    selected_sensor_features,
    selected_survey_features,
    y_all,
):
    # Prepare data for Bayesian Network structure learning
    # Combine all selected features (without UMAP) + target variables

    # Helper function to sanitize column names for pgmpy compatibility
    # (@ and / characters cause parsing issues in pgmpy's formula evaluation)
    def _sanitize_column_name(name):
        return name.replace("@", "_").replace("/", "_")

    # Build feature names
    _raw_feature_names = (
        list(selected_survey_features)
        + [f"{f}_mean" for f in selected_diary_features]
        + [f"{f}_std" for f in selected_diary_features]
        + [f"{f}_mean" for f in selected_sensor_features]
        + [f"{f}_std" for f in selected_sensor_features]
    )

    # Sanitize feature names for pgmpy
    bbn_feature_names = [_sanitize_column_name(f) for f in _raw_feature_names]

    # Combine feature matrices (using scaled versions without UMAP)
    _X_bbn_combined = np.hstack([X_survey_scaled, X_diary_scaled, X_sensor_scaled])

    # Combine with targets
    _bbn_all_data = np.hstack([_X_bbn_combined, y_all])
    bbn_all_columns = bbn_feature_names + list(TARGET_LABELS)

    # Create DataFrame for pgmpy
    bbn_df = pd.DataFrame(_bbn_all_data, columns=bbn_all_columns)

    # Clean any remaining NaN/inf values
    bbn_df = bbn_df.replace([np.inf, -np.inf], np.nan)
    bbn_df = bbn_df.fillna(bbn_df.mean())

    print(f"BBN DataFrame shape: {bbn_df.shape}")
    print(f"Features: {len(bbn_feature_names)}, Targets: {len(TARGET_LABELS)}")
    print(f"Total variables: {len(bbn_all_columns)}")
    return bbn_all_columns, bbn_df


@app.cell
def _(BICGauss, GES, bbn_df, mo):
    # Run GES algorithm for structure learning
    print("Running Greedy Equivalence Search (GES)...")

    try:
        _ges_estimator = GES(bbn_df)
        _learned_model = _ges_estimator.estimate(scoring_method=BICGauss(bbn_df))
        learned_edges = list(_learned_model.edges())

        print(f"GES completed! Learned {len(learned_edges)} edges.")

        if len(learned_edges) > 0:
            print("\nLearned edges:")
            for _edge in learned_edges:
                print(f"  {_edge[0]} -> {_edge[1]}")
        else:
            print("\nNo edges learned - DAG is empty.")
            print("This may indicate insufficient sample size or weak relationships.")

        _ges_success = True
    except Exception as e:
        print(f"GES failed with error: {e}")
        learned_edges = []
        _ges_success = False

    if not _ges_success:
        mo.callout(
            mo.md(
                "**GES algorithm failed.** This may be due to data issues or insufficient samples."
            ),
            kind="danger",
        )
    return (learned_edges,)


@app.cell
def _(
    TARGET_LABELS,
    bbn_all_columns,
    go,
    learned_edges,
    nx,
    selected_diary_features,
    selected_survey_features,
):
    # Visualize the learned DAG
    def _get_node_category_color(node_name):
        """Return color based on variable category."""
        if node_name in TARGET_LABELS:
            return "#e74c3c"  # Red - Target variables
        elif node_name in selected_survey_features:
            return "#3498db"  # Blue - Survey/Demographics
        elif (
            any(node_name.startswith(f"{f}_") for f in selected_diary_features)
            or node_name in selected_diary_features
        ):
            return "#2ecc71"  # Green - Sleep diary
        else:
            return "#f39c12"  # Orange - Sensor/HRV

    # Build networkx graph
    _G_dag = nx.DiGraph()
    _G_dag.add_nodes_from(bbn_all_columns)
    _G_dag.add_edges_from(learned_edges)

    # Only visualize if we have edges
    if len(learned_edges) > 0:
        # Get node colors
        _node_colors = [_get_node_category_color(node) for node in _G_dag.nodes()]

        # Use spring layout
        _pos = nx.spring_layout(_G_dag, k=2, iterations=50, seed=42)

        # Create edge traces
        _edge_x = []
        _edge_y = []
        for _edge in _G_dag.edges():
            _x0, _y0 = _pos[_edge[0]]
            _x1, _y1 = _pos[_edge[1]]
            _edge_x.extend([_x0, _x1, None])
            _edge_y.extend([_y0, _y1, None])

        _edge_trace = go.Scatter(
            x=_edge_x,
            y=_edge_y,
            line=dict(width=2, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        # Create node traces
        _node_x = [_pos[node][0] for node in _G_dag.nodes()]
        _node_y = [_pos[node][1] for node in _G_dag.nodes()]

        _node_trace = go.Scatter(
            x=_node_x,
            y=_node_y,
            mode="markers+text",
            hoverinfo="text",
            text=list(_G_dag.nodes()),
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(
                size=20,
                color=_node_colors,
                line=dict(width=2, color="white"),
            ),
        )

        # Create figure
        _fig_dag = go.Figure(
            data=[_edge_trace, _node_trace],
            layout=go.Layout(
                title=f"Learned Causal DAG (GES Algorithm) - {len(learned_edges)} edges",
                showlegend=False,
                hovermode="closest",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600,
                annotations=[
                    dict(
                        text="Node colors: Blue=Survey, Green=Diary, Orange=Sensor, Red=Target",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=-0.05,
                        font=dict(size=12),
                    )
                ],
            ),
        )
    else:
        # Empty DAG - show message
        _fig_dag = go.Figure()
        _fig_dag.add_annotation(
            text="No edges learned by GES algorithm.<br>The DAG is empty.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        _fig_dag.update_layout(
            title="Learned Causal DAG (GES Algorithm) - Empty",
            height=400,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

    _fig_dag
    return


@app.cell
def _(TARGET_LABELS, learned_edges, mo, pd):
    # Analyze edges involving target variables
    _target_edges = []
    for _edge in learned_edges:
        if _edge[0] in TARGET_LABELS or _edge[1] in TARGET_LABELS:
            _direction = "causes" if _edge[1] in TARGET_LABELS else "is caused by"
            _target_edges.append(
                {
                    "Edge": f"{_edge[0]} -> {_edge[1]}",
                    "Interpretation": f"{_edge[0]} {_direction} {_edge[1]}",
                }
            )

    if len(_target_edges) > 0:
        _target_edges_df = pd.DataFrame(_target_edges)
        mo.vstack(
            [
                mo.md("### Edges Involving Mental Health Targets"),
                mo.md(
                    "These edges suggest potential causal relationships with depression, anxiety, and insomnia scores:"
                ),
                mo.ui.table(_target_edges_df, selection=None),
            ]
        )
    else:
        mo.callout(
            mo.md(
                "**No edges involving target variables were learned.** This suggests the algorithm did not find strong enough evidence for direct causal links between features and mental health outcomes with the current sample size."
            ),
            kind="info",
        )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.1 BBN Predictive Evaluation

    We evaluate the learned DAG by fitting a Linear Gaussian model for each target variable using its parent nodes as predictors. This uses Leave-One-Out Cross-Validation for consistency with baseline models.
    """)
    return


@app.cell
def _(
    LeaveOneOut,
    LinearRegression,
    TARGET_LABELS,
    bbn_df,
    cross_val_predict,
    learned_edges,
    mean_absolute_error,
    np,
    pd,
    r2_score,
):
    # Helper function to get parent nodes (local to this cell)
    def _get_parents_from_edges(edges, node):
        """Get parent nodes for a given node in a DAG."""
        return [edge[0] for edge in edges if edge[1] == node]

    # Evaluate BBN using LOO-CV
    _bbn_results = []

    for _target in TARGET_LABELS:
        _parents = _get_parents_from_edges(learned_edges, _target)
        _y_target = bbn_df[_target].values

        if len(_parents) > 0:
            # Use parents as predictors
            _X_parents = bbn_df[_parents].values

            # LOO-CV prediction
            _loo = LeaveOneOut()
            _model = LinearRegression()
            try:
                _predictions = cross_val_predict(_model, _X_parents, _y_target, cv=_loo)

                # Compute metrics
                _mae = mean_absolute_error(_y_target, _predictions)
                _y_range = np.max(_y_target) - np.min(_y_target)
                _nmae = _mae / _y_range if _y_range > 0 else 0
                _r2 = r2_score(_y_target, _predictions)
                _corr = np.corrcoef(_y_target, _predictions)[0, 1]

                _bbn_results.append(
                    {
                        "Target": _target,
                        "Parents": ", ".join(_parents),
                        "Num_Parents": len(_parents),
                        "MAE": _mae,
                        "NMAE": _nmae,
                        "R2": _r2,
                        "Correlation": _corr if not np.isnan(_corr) else 0,
                    }
                )
            except Exception as _e:
                print(f"Error evaluating {_target}: {_e}")
                _bbn_results.append(
                    {
                        "Target": _target,
                        "Parents": ", ".join(_parents),
                        "Num_Parents": len(_parents),
                        "MAE": np.nan,
                        "NMAE": np.nan,
                        "R2": np.nan,
                        "Correlation": np.nan,
                    }
                )
        else:
            # No parents - predict using mean (baseline)
            _mean_pred = np.mean(_y_target)
            _predictions = np.full_like(_y_target, _mean_pred)
            _mae = mean_absolute_error(_y_target, _predictions)
            _y_range = np.max(_y_target) - np.min(_y_target)
            _nmae = _mae / _y_range if _y_range > 0 else 0

            _bbn_results.append(
                {
                    "Target": _target,
                    "Parents": "(none - using mean)",
                    "Num_Parents": 0,
                    "MAE": _mae,
                    "NMAE": _nmae,
                    "R2": 0.0,
                    "Correlation": 0.0,
                }
            )

    bbn_results_df = pd.DataFrame(_bbn_results)
    print("BBN Evaluation Complete!")
    return (bbn_results_df,)


@app.cell
def _(bbn_results_df, mo):
    # Display BBN results
    mo.vstack(
        [
            mo.md("### BBN Predictive Performance (LOO-CV)"),
            mo.ui.table(bbn_results_df.round(4), selection=None),
        ]
    )
    return


@app.cell
def _(bbn_results_df, mo, pd, results_df):
    # Compare BBN vs Baseline models
    # Get best baseline results for each target
    _baseline_comparison = []

    for _, _bbn_row in bbn_results_df.iterrows():
        _target = _bbn_row["Target"]
        # Find best baseline R2 for this target
        _target_baseline = results_df[results_df["Target"] == _target]
        if len(_target_baseline) > 0:
            _best_baseline_r2 = _target_baseline["R2"].max()
            _best_baseline_config = _target_baseline.loc[
                _target_baseline["R2"].idxmax(), "Configuration"
            ]
        else:
            _best_baseline_r2 = 0
            _best_baseline_config = "N/A"

        _baseline_comparison.append(
            {
                "Target": _target,
                "BBN R2": _bbn_row["R2"],
                "Best Baseline R2": _best_baseline_r2,
                "Best Baseline Config": _best_baseline_config,
                "BBN Better": _bbn_row["R2"] > _best_baseline_r2,
            }
        )

    _comparison_df = pd.DataFrame(_baseline_comparison)

    mo.vstack(
        [
            mo.md("### BBN vs Baseline Model Comparison"),
            mo.ui.table(_comparison_df.round(4), selection=None),
        ]
    )
    return


@app.cell
def _(learned_edges, mo):
    # Interpretation section
    _n_edges = len(learned_edges)

    if _n_edges > 0:
        mo.md(f"""
    ### 4.2 Interpretation of Learned Causal Structure

    The GES algorithm discovered **{_n_edges} directed edges** in the causal graph. Key observations:

    **What the DAG Reveals:**
    1. **Direct Influences**: Edges pointing TO target variables (PHQ9_F, GAD7_F, ISI_F) suggest features that may directly influence mental health outcomes
    2. **Indirect Pathways**: Features connected through intermediate nodes may have indirect effects
    3. **Independence**: Variables without edges are conditionally independent given their parents

    **Causal vs Correlation:**
    - Unlike the correlation heatmap (Section 2.2), the DAG shows *directed* relationships
    - An edge A -> B means "knowing A helps predict B, even after accounting for other parents of B"
    - Missing edges between correlated variables suggest their correlation is explained by common causes

    **Limitations:**
    - With n=49 samples, the algorithm may miss true causal links (false negatives)
    - Some edges may be spurious due to limited data (false positives)
    - DAG structure represents *statistical* dependencies, not necessarily true causation
    - Unmeasured confounders could explain apparent causal relationships

    **Clinical Implications:**
    The learned structure can help prioritize which features to target for intervention studies. Features with direct edges to mental health targets are prime candidates for further investigation.
        """)
    else:
        mo.md("""
    ### 4.2 Interpretation of Empty DAG

    The GES algorithm did not learn any edges, resulting in an empty DAG. This outcome suggests:

    1. **Weak Signal**: The relationships between features and mental health outcomes may be too weak to detect with n=49 samples
    2. **High Dimensionality**: With many features relative to samples, the algorithm becomes conservative to avoid false positives
    3. **BIC Penalty**: The Bayesian Information Criterion penalizes model complexity, preferring simpler (empty) models when evidence is weak

    **Recommendations:**
    - Reduce the number of features using the Feature Selection controls above
    - Focus on features with strongest correlations from Section 2.2
    - Consider that mental health outcomes may be influenced by factors not captured in sensor/diary data

    This result is consistent with the baseline model findings showing weak predictive performance.
        """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 5. Model Comparison Visualizations
    """)
    return


@app.cell
def _(TARGET_LABELS, TARGET_NAMES, go, make_subplots, results_df):
    # Create grouped bar chart for R² comparison
    fig_r2_comparison = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[TARGET_NAMES[t] for t in TARGET_LABELS],
        shared_yaxes=True,
    )

    config_names = ["Survey + Diary", "Survey + Sensor", "Survey + Diary + Sensor"]
    bar_colors = {"Ridge": "#636EFA", "Lasso": "#EF553B"}

    for _col_idx, _target in enumerate(TARGET_LABELS):
        _target_data = results_df[results_df["Target"] == _target]

        for _reg_type in ["Ridge", "Lasso"]:
            _reg_data = _target_data[_target_data["Regularization"] == _reg_type]

            fig_r2_comparison.add_trace(
                go.Bar(
                    name=_reg_type,
                    x=_reg_data["Configuration"],
                    y=_reg_data["R2"],
                    marker_color=bar_colors[_reg_type],
                    showlegend=(_col_idx == 0),
                    text=_reg_data["R2"].round(3),
                    textposition="outside",
                ),
                row=1,
                col=_col_idx + 1,
            )

    fig_r2_comparison.update_layout(
        title_text="R² Score Comparison Across Configurations",
        barmode="group",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_r2_comparison.update_yaxes(title_text="R² Score", row=1, col=1)
    fig_r2_comparison.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_r2_comparison
    return


@app.cell
def _(TARGET_LABELS, go, results_df):
    # Create heatmap of R² scores
    pivot_r2 = results_df.pivot_table(
        values="R2", index=["Configuration", "Regularization"], columns="Target"
    )[TARGET_LABELS]  # Ensure column order

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=pivot_r2.values,
            x=[f"{t}" for t in TARGET_LABELS],
            y=[f"{idx[0]}<br>({idx[1]})" for idx in pivot_r2.index],
            colorscale="RdYlGn",
            zmid=0,
            text=pivot_r2.values.round(3),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="R²"),
        )
    )

    fig_heatmap.update_layout(
        title="R² Score Heatmap by Configuration and Target",
        xaxis_title="Target Variable",
        yaxis_title="Model Configuration",
        height=400,
    )
    fig_heatmap
    return


@app.cell
def _(TARGET_LABELS, go, make_subplots, results_df):
    # MAE comparison
    fig_mae = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[f"{t}" for t in TARGET_LABELS],
        shared_yaxes=False,
    )

    for _col_idx_mae, _target_mae in enumerate(TARGET_LABELS):
        _target_data_mae = results_df[results_df["Target"] == _target_mae]

        for _reg_type_mae in ["Ridge", "Lasso"]:
            _reg_data_mae = _target_data_mae[
                _target_data_mae["Regularization"] == _reg_type_mae
            ]

            fig_mae.add_trace(
                go.Bar(
                    name=_reg_type_mae,
                    x=_reg_data_mae["Configuration"],
                    y=_reg_data_mae["MAE"],
                    marker_color="#636EFA" if _reg_type_mae == "Ridge" else "#EF553B",
                    showlegend=(_col_idx_mae == 0),
                    text=_reg_data_mae["MAE"].round(2),
                    textposition="outside",
                ),
                row=1,
                col=_col_idx_mae + 1,
            )

    fig_mae.update_layout(
        title_text="Mean Absolute Error (MAE) Comparison",
        barmode="group",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_mae.update_yaxes(title_text="MAE", row=1, col=1)
    fig_mae
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 6. Predicted vs Actual Analysis
    """)
    return


@app.cell
def _(TARGET_LABELS, mo, results_df):
    # Selection controls for predicted vs actual plot
    best_config = results_df.loc[results_df["R2"].idxmax()]

    pred_config_selector = mo.ui.dropdown(
        options=["Survey + Diary", "Survey + Sensor", "Survey + Diary + Sensor"],
        value=best_config["Configuration"],
        label="Configuration",
    )

    pred_reg_selector = mo.ui.dropdown(
        options=["Ridge", "Lasso"],
        value=best_config["Regularization"],
        label="Regularization",
    )

    pred_target_selector = mo.ui.dropdown(
        options=TARGET_LABELS, value=best_config["Target"], label="Target"
    )

    mo.hstack([pred_config_selector, pred_reg_selector, pred_target_selector], gap=2)
    return pred_config_selector, pred_reg_selector, pred_target_selector


@app.cell
def _(
    TARGET_NAMES,
    go,
    np,
    pred_config_selector,
    pred_reg_selector,
    pred_target_selector,
    predictions_store,
    user_ids,
):
    # Predicted vs Actual scatter plot
    _key = (
        pred_config_selector.value,
        pred_reg_selector.value,
        pred_target_selector.value,
    )
    _data = predictions_store[_key]
    _preds = _data["predictions"]
    _actuals = _data["actuals"]

    # Compute trendline
    _z = np.polyfit(_actuals, _preds, 1)
    _p = np.poly1d(_z)
    _x_line = np.linspace(_actuals.min(), _actuals.max(), 100)

    fig_pred_actual = go.Figure()

    # Scatter points
    fig_pred_actual.add_trace(
        go.Scatter(
            x=_actuals,
            y=_preds,
            mode="markers",
            marker=dict(size=10, color="#636EFA", opacity=0.7),
            name="Predictions",
            text=user_ids,
            hovertemplate="User: %{text}<br>Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>",
        )
    )

    # Perfect prediction line (y=x)
    fig_pred_actual.add_trace(
        go.Scatter(
            x=[_actuals.min(), _actuals.max()],
            y=[_actuals.min(), _actuals.max()],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Perfect Prediction",
        )
    )

    # Regression trendline
    fig_pred_actual.add_trace(
        go.Scatter(
            x=_x_line,
            y=_p(_x_line),
            mode="lines",
            line=dict(color="red", width=2),
            name="Trendline",
        )
    )

    fig_pred_actual.update_layout(
        title=f"Predicted vs Actual: {TARGET_NAMES[pred_target_selector.value]}<br><sub>{pred_config_selector.value} + {pred_reg_selector.value}</sub>",
        xaxis_title="Actual Score",
        yaxis_title="Predicted Score",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig_pred_actual
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 6.1 Residual Analysis

    Understanding prediction errors helps diagnose model issues and identify hard-to-predict participants.
    """)
    return


@app.cell
def _(
    TARGET_NAMES,
    go,
    make_subplots,
    np,
    pred_config_selector,
    pred_reg_selector,
    pred_target_selector,
    predictions_store,
    user_ids,
):
    # Get predictions for selected configuration
    _key = (
        pred_config_selector.value,
        pred_reg_selector.value,
        pred_target_selector.value,
    )
    _data = predictions_store[_key]
    _preds = _data["predictions"]
    _actuals = _data["actuals"]
    residuals = _actuals - _preds

    # Create subplot: residual histogram + residual vs predicted
    fig_residuals = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Residual Distribution", "Residuals vs Predicted"],
    )

    # Histogram of residuals
    fig_residuals.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=15,
            name="Residuals",
            marker_color="#636EFA",
            opacity=0.75,
        ),
        row=1,
        col=1,
    )

    # Residual vs Predicted scatter
    fig_residuals.add_trace(
        go.Scatter(
            x=_preds,
            y=residuals,
            mode="markers",
            name="Residuals",
            marker=dict(size=10, color="#636EFA", opacity=0.7),
            text=user_ids,
            hovertemplate="User: %{text}<br>Predicted: %{x:.1f}<br>Residual: %{y:.1f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Add zero line
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

    fig_residuals.update_layout(
        title=f"Residual Analysis: {TARGET_NAMES[pred_target_selector.value]}<br><sub>{pred_config_selector.value} + {pred_reg_selector.value}</sub>",
        height=400,
        showlegend=False,
    )
    fig_residuals.update_xaxes(title_text="Residual (Actual - Predicted)", row=1, col=1)
    fig_residuals.update_yaxes(title_text="Count", row=1, col=1)
    fig_residuals.update_xaxes(title_text="Predicted Score", row=1, col=2)
    fig_residuals.update_yaxes(title_text="Residual", row=1, col=2)

    # Compute residual statistics
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)

    fig_residuals
    return (residuals,)


@app.cell
def _(
    mo,
    np,
    pd,
    pred_config_selector,
    pred_reg_selector,
    pred_target_selector,
    predictions_store,
    residuals,
    user_ids,
):
    # Table of worst predictions
    _key = (
        pred_config_selector.value,
        pred_reg_selector.value,
        pred_target_selector.value,
    )
    _data = predictions_store[_key]
    _preds = _data["predictions"]
    _actuals = _data["actuals"]

    error_df = pd.DataFrame(
        {
            "User": user_ids,
            "Actual": _actuals,
            "Predicted": np.round(_preds, 2),
            "Error": np.round(residuals, 2),
            "Abs_Error": np.round(np.abs(residuals), 2),
        }
    ).sort_values("Abs_Error", ascending=False)

    mo.vstack(
        [
            mo.md("**Worst Predictions (highest absolute error):**"),
            mo.md(
                "_These participants are hardest for the model to predict accurately._"
            ),
            mo.ui.table(error_df.head(10), selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 7. Patient Time Series Trends

    Explore individual patient sensor and sleep diary data over time.
    """)
    return


@app.cell
def _(mo, user_ids):
    # Patient selector
    patient_selector = mo.ui.dropdown(
        options=user_ids, value=user_ids[0], label="Select Patient"
    )
    patient_selector
    return (patient_selector,)


@app.cell
def _(DIARY_FEATURE_COLS, df_diary, go, make_subplots, patient_selector, pd):
    # Patient sleep diary trends
    _patient_id = patient_selector.value
    _patient_diary = df_diary[df_diary["userId"] == _patient_id].copy()

    fig_patient_diary = None
    if len(_patient_diary) > 0:
        _patient_diary["date"] = pd.to_datetime(_patient_diary["date"])
        _patient_diary = _patient_diary.sort_values("date")

        fig_patient_diary = make_subplots(
            rows=2, cols=3, subplot_titles=DIARY_FEATURE_COLS
        )

        for _i, _col in enumerate(DIARY_FEATURE_COLS):
            _row = _i // 3 + 1
            _col_idx = _i % 3 + 1

            fig_patient_diary.add_trace(
                go.Scatter(
                    x=_patient_diary["date"],
                    y=_patient_diary[_col],
                    mode="lines+markers",
                    name=_col,
                    line=dict(width=2),
                    marker=dict(size=6),
                ),
                row=_row,
                col=_col_idx,
            )

        fig_patient_diary.update_layout(
            title=f"Sleep Diary Trends - Patient: {_patient_id}",
            height=500,
            showlegend=False,
        )
    else:
        print(f"No diary data available for patient {_patient_id}")
    fig_patient_diary
    return


@app.cell
def _(SENSOR_FEATURE_COLS, df_sensor, go, make_subplots, patient_selector, pd):
    # Patient sensor HRV trends (aggregated by day)
    _patient_id_sensor = patient_selector.value
    _patient_sensor = df_sensor[df_sensor["deviceId"] == _patient_id_sensor].copy()

    fig_patient_sensor = None
    if len(_patient_sensor) > 0:
        # Convert timestamp to datetime and aggregate by day
        _patient_sensor["datetime"] = pd.to_datetime(
            _patient_sensor["ts_start"], unit="ms"
        )
        _patient_sensor["date"] = _patient_sensor["datetime"].dt.date

        # Daily aggregation
        _daily_sensor = (
            _patient_sensor.groupby("date")[SENSOR_FEATURE_COLS].mean().reset_index()
        )
        _daily_sensor["date"] = pd.to_datetime(_daily_sensor["date"])

        # Select key HRV features for visualization
        _key_features = ["HR", "rmssd", "sdnn", "lf/hf", "steps", "light_avg"]

        fig_patient_sensor = make_subplots(rows=2, cols=3, subplot_titles=_key_features)

        _colors_sensor = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
        ]

        for _i, _feat in enumerate(_key_features):
            _row_s = _i // 3 + 1
            _col_s = _i % 3 + 1

            fig_patient_sensor.add_trace(
                go.Scatter(
                    x=_daily_sensor["date"],
                    y=_daily_sensor[_feat],
                    mode="lines+markers",
                    name=_feat,
                    line=dict(width=2, color=_colors_sensor[_i]),
                    marker=dict(size=5),
                ),
                row=_row_s,
                col=_col_s,
            )

        fig_patient_sensor.update_layout(
            title=f"Sensor HRV Daily Trends - Patient: {_patient_id_sensor}",
            height=500,
            showlegend=False,
        )
    else:
        print(f"No sensor data available for patient {_patient_id_sensor}")
    fig_patient_sensor
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 8. Best Model Summary
    """)
    return


@app.cell
def _(TARGET_LABELS, TARGET_NAMES, mo, pd, results_df):
    # Find best model for each target
    best_models = []
    for _target in TARGET_LABELS:
        _target_results = results_df[results_df["Target"] == _target]
        _best_idx = _target_results["R2"].idxmax()
        _best = _target_results.loc[_best_idx]
        best_models.append(
            {
                "Target": TARGET_NAMES[_target],
                "Best Configuration": _best["Configuration"],
                "Regularization": _best["Regularization"],
                "R²": round(_best["R2"], 4),
                "MAE": round(_best["MAE"], 3),
                "RMSE": round(_best["RMSE"], 3),
                "Features": _best["Features"],
            }
        )

    best_models_df = pd.DataFrame(best_models)

    mo.vstack(
        [
            mo.md("### Best Model for Each Clinical Target"),
            mo.ui.table(best_models_df, selection=None),
        ]
    )
    return


@app.cell
def _(alpha, mo, sensor_label, umap_enabled, umap_n_components):
    mo.md(f"""
    ### Key Findings

    **Current Settings:**
    - Regularization alpha: {alpha:.4f}
    - UMAP enabled: {umap_enabled.value}
    - UMAP components: {umap_n_components.value if umap_enabled.value else "N/A"}
    - Sensor feature representation: {sensor_label}

    **Observations:**
    1. **Survey + Diary** configuration generally performs best for anxiety (GAD-7) prediction
    2. **UMAP dimensionality reduction** helps mitigate overfitting from high-dimensional sensor features
    3. **StandardScaler normalization** after feature aggregation improves model stability
    4. Small sample size (n=49) remains a fundamental challenge for all configurations

    **Next Steps:**
    - Explore classification approach with binary PHQ-9 binning (depressed vs. non-depressed)
    - Investigate Bayesian Belief Networks
    - Evaluate time-series specific models (LSTM, Transformers)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 8.1 Why Is Performance Poor?

    A comprehensive summary of factors limiting model performance.
    """)
    return


@app.cell
def _(X_all, mo):
    n_samples = X_all.shape[0]
    n_features = X_all.shape[1]
    ratio = n_features / n_samples if n_samples > 0 else 0

    mo.callout(
        mo.md(f"""
    **Root Causes of Poor Performance:**

    1. **Insufficient Sample Size**: n={n_samples} participants with {n_features} features
       - Rule of thumb: need 10-20 samples per feature for reliable regression
       - Current ratio: **{ratio:.2f} features per sample** (ideal: < 0.1)
       - This means we need **{n_features * 10}-{n_features * 20} samples** for reliable results

    2. **Severe Class Imbalance**: 
       - 63% minimal depression (PHQ-9 <= 4), only 8% moderate+ (PHQ-9 >= 10)
       - 80% minimal anxiety (GAD-7 <= 4), only 2% moderate+ (GAD-7 >= 10)
       - Model learns to predict "average" (low scores) for everyone

    3. **Data Quality Issues**:
       - Corrupted `lf`, `hf` features (values up to 1e+46)
       - 55% missing data in `steps`, `calories`, `distance`
       - Use feature selection above to exclude problematic features

    4. **Low Signal-to-Noise Ratio**:
       - Weak correlations between features and targets (see correlation heatmap)
       - High within-group variability relative to between-group differences

    **Recommendations:**
    - Use feature selection to reduce dimensionality (aim for < 10 features)
    - Consider binary classification (depressed vs not) instead of regression
    - Collect more participants, especially with clinical symptoms
    - Focus on the most reliable features (HR, rmssd, sdnn, sleep metrics)
        """),
        kind="info",
    )
    return


@app.cell
def _(LeaveOneOut, Ridge, X_all, cross_val_predict, go, np, r2_score, y_all):
    # Learning curve - subsample to show effect of sample size
    sample_sizes = [10, 15, 20, 25, 30, 35, 40, 45, min(49, len(y_all))]
    r2_scores_by_size = []
    r2_std_by_size = []

    for size in sample_sizes:
        if size > len(y_all):
            continue
        # Random subsample (average over multiple trials)
        r2_trials = []
        for trial in range(20):
            np.random.seed(trial)
            idx = np.random.choice(len(y_all), size, replace=False)
            X_sub, y_sub = X_all[idx], y_all[idx, 0]  # PHQ9

            if X_sub.shape[1] > 0 and size > 3:
                try:
                    loo = LeaveOneOut()
                    preds = cross_val_predict(Ridge(alpha=1.0), X_sub, y_sub, cv=loo)
                    r2 = r2_score(y_sub, preds)
                    if not np.isnan(r2) and not np.isinf(r2):
                        r2_trials.append(r2)
                except:
                    pass

        if len(r2_trials) > 0:
            r2_scores_by_size.append(np.mean(r2_trials))
            r2_std_by_size.append(np.std(r2_trials))
        else:
            r2_scores_by_size.append(np.nan)
            r2_std_by_size.append(0)

    # Filter valid results
    valid_sizes = [
        s
        for s, r in zip(sample_sizes[: len(r2_scores_by_size)], r2_scores_by_size)
        if not np.isnan(r)
    ]
    valid_r2 = [r for r in r2_scores_by_size if not np.isnan(r)]
    valid_std = [
        s for r, s in zip(r2_scores_by_size, r2_std_by_size) if not np.isnan(r)
    ]

    if len(valid_sizes) > 0:
        fig_learning = go.Figure()

        # Main line
        fig_learning.add_trace(
            go.Scatter(
                x=valid_sizes,
                y=valid_r2,
                mode="lines+markers",
                name="Mean R²",
                line=dict(color="#636EFA", width=2),
                marker=dict(size=8),
            )
        )

        # Error band
        upper_bound = [r + s for r, s in zip(valid_r2, valid_std)]
        lower_bound = [r - s for r, s in zip(valid_r2, valid_std)]

        fig_learning.add_trace(
            go.Scatter(
                x=valid_sizes + valid_sizes[::-1],
                y=upper_bound + lower_bound[::-1],
                fill="toself",
                fillcolor="rgba(99, 110, 250, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Std Dev",
            )
        )

        fig_learning.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            annotation_text="R²=0 (no better than mean)",
        )

        fig_learning.update_layout(
            title="Learning Curve: R² vs Sample Size (PHQ-9, Ridge Regression)",
            xaxis_title="Number of Samples",
            yaxis_title="R² Score (LOO-CV)",
            height=400,
            showlegend=True,
        )
        fig_learning
    else:
        print("Cannot generate learning curve - insufficient data")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 9. Export Results
    """)
    return


@app.cell
def _(OUTPUT_DIR, os, results_df):
    # Save results to CSV
    output_path = os.path.join(OUTPUT_DIR, "final_model_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    return


if __name__ == "__main__":
    app.run()
