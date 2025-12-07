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
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline

    # UMAP for dimensionality reduction
    import umap

    # Plotly for interactive visualizations
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    print("All imports successful!")
    return (
        Lasso,
        LeaveOneOut,
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
        "height",
        "weight",
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
            mo.ui.table(_selected_df.head(20), selection=None),
        ]
    )
    return


@app.cell
def _(TARGET_LABELS, TARGET_NAMES, df_survey, go, make_subplots):
    # Create score distribution plots
    fig_distributions = make_subplots(
        rows=1, cols=3, subplot_titles=[TARGET_NAMES[t] for t in TARGET_LABELS]
    )

    colors = ["#636EFA", "#EF553B", "#00CC96"]

    for _i, _target in enumerate(TARGET_LABELS):
        fig_distributions.add_trace(
            go.Histogram(
                x=df_survey[_target],
                name=TARGET_NAMES[_target],
                marker_color=colors[_i],
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
    def aggregate_sequences(seq_data: dict, feature_cols: list) -> np.ndarray:
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
def _(
    DIARY_FEATURE_COLS,
    SENSOR_FEATURE_COLS,
    SURVEY_FEATURE_COLS,
    TARGET_LABELS,
    aggregate_sequences,
    df_diary,
    df_sensor,
    df_survey,
    prepare_sequence_data,
):
    # Prepare data for all users
    # Get intersection of users across all datasets
    survey_users = set(df_survey["deviceId"].unique())
    diary_users = set(df_diary["userId"].unique())
    sensor_users = set(df_sensor["deviceId"].unique())

    common_users = sorted(survey_users & diary_users & sensor_users)
    n_users = len(common_users)
    print(f"Common users across all datasets: {n_users}")

    # Prepare survey features and labels
    survey_data = df_survey[df_survey["deviceId"].isin(common_users)].sort_values(
        "deviceId"
    )
    X_survey_raw = (
        survey_data[SURVEY_FEATURE_COLS]
        .fillna(survey_data[SURVEY_FEATURE_COLS].mean())
        .values
    )
    y_all = survey_data[TARGET_LABELS].fillna(0).values
    user_ids = survey_data["deviceId"].tolist()

    # Prepare diary sequence data
    diary_seq = prepare_sequence_data(
        df_diary[df_diary["userId"].isin(common_users)], "userId", DIARY_FEATURE_COLS
    )
    X_diary_raw = aggregate_sequences(diary_seq, DIARY_FEATURE_COLS)

    # Prepare sensor sequence data
    sensor_seq = prepare_sequence_data(
        df_sensor[df_sensor["deviceId"].isin(common_users)],
        "deviceId",
        SENSOR_FEATURE_COLS,
    )
    X_sensor_raw = aggregate_sequences(sensor_seq, SENSOR_FEATURE_COLS)

    # Print shapes
    print(f"\nFeature dimensions (before scaling):")
    print(f"  Survey: {X_survey_raw.shape}")
    print(f"  Diary (aggregated): {X_diary_raw.shape}")
    print(f"  Sensor (aggregated): {X_sensor_raw.shape}")
    print(f"  Labels: {y_all.shape}")
    return X_diary_raw, X_sensor_raw, X_survey_raw, user_ids, y_all


@app.cell
def _(mo):
    # Interactive feature selection
    mo.md("""
    ### Feature Selection
    """)
    return


@app.cell
def _(DIARY_FEATURE_COLS, SENSOR_FEATURE_COLS, SURVEY_FEATURE_COLS):
    # Create feature options with source labels
    survey_options = {f"[Survey] {f}": f for f in SURVEY_FEATURE_COLS}
    diary_options = {f"[Diary] {f}_mean": f"{f}_mean" for f in DIARY_FEATURE_COLS}
    diary_options.update({f"[Diary] {f}_std": f"{f}_std" for f in DIARY_FEATURE_COLS})
    sensor_options = {f"[Sensor] {f}_mean": f"{f}_mean" for f in SENSOR_FEATURE_COLS}
    sensor_options.update(
        {f"[Sensor] {f}_std": f"{f}_std" for f in SENSOR_FEATURE_COLS}
    )
    return


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
    # Apply StandardScaler to all features
    scaler_survey = StandardScaler()
    scaler_diary = StandardScaler()
    scaler_sensor = StandardScaler()

    X_survey_scaled = scaler_survey.fit_transform(X_survey_raw)
    X_diary_scaled = scaler_diary.fit_transform(X_diary_raw)
    X_sensor_scaled = scaler_sensor.fit_transform(X_sensor_raw)

    # Handle NaN values after scaling (can occur with zero-variance columns)
    X_survey_scaled = np.nan_to_num(X_survey_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_diary_scaled = np.nan_to_num(X_diary_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_sensor_scaled = np.nan_to_num(X_sensor_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply UMAP to sensor features if enabled
    if umap_enabled.value:
        umap_model = umap.UMAP(
            n_components=umap_n_components.value,
            n_neighbors=umap_n_neighbors.value,
            min_dist=umap_min_dist.value,
            random_state=42,
        )
        X_sensor_reduced = umap_model.fit_transform(X_sensor_scaled)
        sensor_label = f"Sensor (UMAP {umap_n_components.value}D)"
    else:
        X_sensor_reduced = X_sensor_scaled
        sensor_label = "Sensor (scaled)"

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
        X_sensor_reduced,
        X_survey_diary,
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
    ---
    ## 4. Model Comparison Visualizations
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
    ## 5. Predicted vs Actual Analysis
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
    ---
    ## 6. Patient Time Series Trends

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
    ## 7. Best Model Summary
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
    ---
    ## 8. Export Results
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
