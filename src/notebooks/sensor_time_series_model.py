# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "pandas",
#     "numpy",
#     "scikit-learn",
#     "plotly",
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
def _():
    import copy
    import os
    import random
    import sys
    import warnings

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import torch
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from torch import nn
    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
    from torch.utils.data import DataLoader, Dataset, Sampler

    warnings.filterwarnings("ignore")
    torch.manual_seed(42)
    np.random.seed(42)
    return (
        DataLoader,
        Dataset,
        Sampler,
        copy,
        go,
        mean_absolute_error,
        mean_squared_error,
        nn,
        np,
        os,
        pack_padded_sequence,
        pad_sequence,
        pd,
        r2_score,
        random,
        sys,
        torch,
        train_test_split,
    )


@app.cell
def _(os, sys):
    _current_dir = (
        os.path.dirname(os.path.abspath(__file__))
        if "__file__" in dir()
        else os.getcwd()
    )

    if "notebooks" in _current_dir:
        PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, "..", ".."))
    else:
        PROJECT_ROOT = _current_dir

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from src.utils.base_dataloaders import SensorHRVDataset

    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    SURVEY_PATH = os.path.join(DATA_DIR, "survey.csv")
    SENSOR_PATH = os.path.join(DATA_DIR, "sensor_hrv.csv")

    TARGET_LABELS = ["PHQ9_F", "GAD7_F", "ISI_F"]
    TARGET_NAMES = {
        "PHQ9_F": "Depression (PHQ-9)",
        "GAD7_F": "Anxiety (GAD-7)",
        "ISI_F": "Insomnia Severity (ISI)",
    }

    SENSOR_FEATURE_COLS = [
        "missingness_score",
        "HR",
        "ibi",
        "light_avg",
        "sdnn",
        "sdsd",
        "rmssd",
        "pnn20",
        "pnn50",
    ]

    print(f"Project root: {PROJECT_ROOT}")
    return (
        SENSOR_FEATURE_COLS,
        SENSOR_PATH,
        SURVEY_PATH,
        SensorHRVDataset,
        TARGET_LABELS,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Sensor Sequence Modeling with LSTM / Transformer

    This notebook builds sequence models directly on the high-resolution HRV sensor data.
    We leverage the existing `SensorHRVDataset` to construct per-user sequences and train
    neural regressors (LSTM or Transformer encoder) to predict clinical assessment scores.

    **Workflow**

    1. Load sensor sequences and align them with survey labels.
    2. Split users into train/validation folds.
    3. Train either an LSTM or Transformer model.
    4. Inspect learning curves and prediction quality.
    """)
    return


@app.cell
def _(
    Dataset,
    SENSOR_FEATURE_COLS,
    SensorHRVDataset,
    TARGET_LABELS,
    pad_sequence,
    pd,
    torch,
):
    class SensorSequenceRegressionDataset(Dataset):
        """
        Wraps SensorHRVDataset to attach survey labels per user.
        """

        def __init__(
            self,
            sensor_csv: str,
            survey_csv: str,
            target_label: str,
            feature_cols=None,
            normalize: bool = True,
            min_length: int = 50,
            num_windows: int = 20,
        ):
            if target_label not in TARGET_LABELS:
                raise ValueError(f"Unknown target label: {target_label}")

            survey_df = pd.read_csv(survey_csv)
            label_map = (
                survey_df.set_index("deviceId")[target_label].dropna().to_dict()
            )

            feature_cols = feature_cols or SENSOR_FEATURE_COLS
            base_dataset = SensorHRVDataset(
                csv_file=sensor_csv,
                mode="user",
                feature_cols=feature_cols,
                normalize=normalize,
            )

            self.num_windows = max(int(num_windows), 1)
            self.samples = []
            self.user_ids = []
            self.lengths = []
            for seq, user_id in base_dataset:
                label = label_map.get(user_id)
                if label is None:
                    continue
                if seq.shape[0] < min_length:
                    continue
                compressed_seq = self._compress_sequence(seq)
                self.samples.append((compressed_seq, float(label), user_id))
                self.user_ids.append(user_id)
                self.lengths.append(compressed_seq.shape[0])

            self.feature_dim = len(feature_cols)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            seq, label, user_id = self.samples[idx]
            return seq, torch.tensor(label, dtype=torch.float32), user_id

        def _compress_sequence(self, seq: torch.Tensor) -> torch.Tensor:
            target_len = self.num_windows
            length = seq.shape[0]
            feature_dim = seq.shape[1]

            if length == 0:
                return seq.new_zeros((target_len, feature_dim))

            if length <= target_len:
                if length < target_len:
                    padding = seq.new_zeros((target_len - length, feature_dim))
                    seq = torch.cat([seq, padding], dim=0)
                return seq

            windows = []
            for window_idx in range(target_len):
                start = (window_idx * length) // target_len
                end = ((window_idx + 1) * length) // target_len
                if window_idx == target_len - 1:
                    end = length
                if end <= start:
                    end = min(start + 1, length)
                window_slice = seq[start:end]
                windows.append(window_slice.mean(dim=0, keepdim=True))

            return torch.cat(windows, dim=0)

    def collate_sequences(batch):
        sequences, labels, lengths, ids = [], [], [], []
        for seq, label, user_id in batch:
            sequences.append(seq)
            labels.append(label)
            lengths.append(seq.shape[0])
            ids.append(user_id)

        padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        labels = torch.stack(labels)
        lengths = torch.tensor(lengths, dtype=torch.long)
        return padded, lengths, labels, ids
    return SensorSequenceRegressionDataset, collate_sequences


@app.cell
def _(Sampler, random):
    class BucketBatchSamplerSubset(Sampler):
        """
        Groups indices with similar sequence lengths to minimize padding.
        """

        def __init__(self, indices, lengths, batch_size, shuffle=True):
            self.batch_size = max(int(batch_size), 1)
            self.shuffle = shuffle
            self.lengths = lengths
            self.indices = list(indices)
            self._build_batches()

        def _build_batches(self):
            length_pairs = [(idx, self.lengths[idx]) for idx in self.indices]
            length_pairs.sort(key=lambda x: x[1])
            self.batches = []
            for i in range(0, len(length_pairs), self.batch_size):
                batch = [idx for idx, _ in length_pairs[i : i + self.batch_size]]
                self.batches.append(batch)

        def __iter__(self):
            batches = list(self.batches)
            if self.shuffle:
                random.shuffle(batches)
            for batch in batches:
                yield batch

        def __len__(self):
            return len(self.batches)
    return (BucketBatchSamplerSubset,)


@app.cell
def _(TARGET_LABELS, mo):
    target_selector = mo.ui.dropdown(
        options=[t for t in TARGET_LABELS],
        value="PHQ9_F",
        label="Target Score",
        full_width=True,
    )

    model_selector = mo.ui.dropdown(
        options=["lstm", "transformer"],
        value="lstm",
        label="Model Type",
    )

    hidden_slider = mo.ui.slider(
        start=32,
        stop=256,
        step=32,
        value=128,
        label="Hidden Dimension",
    )

    layers_slider = mo.ui.slider(
        start=1,
        stop=4,
        step=1,
        value=2,
        label="Encoder Layers",
    )

    dropout_slider = mo.ui.slider(
        start=0.0,
        stop=0.5,
        step=0.05,
        value=0.1,
        label="Dropout",
    )

    lr_slider = mo.ui.slider(
        start=-4,
        stop=-2,
        step=0.25,
        value=-3.0,
        label="Learning Rate (log10)",
    )

    epochs_slider = mo.ui.slider(
        start=5,
        stop=40,
        step=5,
        value=15,
        label="Epochs",
    )

    batch_slider = mo.ui.slider(
        start=2,
        stop=16,
        step=2,
        value=4,
        label="Batch Size (users)",
    )

    split_slider = mo.ui.slider(
        start=0.05,
        stop=0.20,
        step=0.05,
        value=0.10,
        label="Validation Fraction",
    )

    min_seq_slider = mo.ui.slider(
        start=50,
        stop=600,
        step=50,
        value=200,
        label="Minimum sequence length (timesteps)",
    )

    num_windows_slider = mo.ui.slider(
        start=5,
        stop=60,
        step=5,
        value=20,
        label="Sequence windows per user",
    )

    train_button = mo.ui.button(label="Train model")

    mo.vstack(
        [
            mo.hstack([target_selector, model_selector]),
            mo.hstack([hidden_slider, layers_slider, dropout_slider], gap=2),
            mo.hstack([lr_slider, epochs_slider, batch_slider], gap=2),
            mo.hstack([split_slider, min_seq_slider, num_windows_slider], gap=2),
            train_button,
        ],
        gap=2,
    )
    return (
        batch_slider,
        dropout_slider,
        epochs_slider,
        hidden_slider,
        layers_slider,
        lr_slider,
        min_seq_slider,
        model_selector,
        num_windows_slider,
        split_slider,
        target_selector,
        train_button,
    )


@app.cell
def _(
    SENSOR_PATH,
    SURVEY_PATH,
    SensorSequenceRegressionDataset,
    min_seq_slider,
    num_windows_slider,
    target_selector,
):
    dataset = SensorSequenceRegressionDataset(
        sensor_csv=SENSOR_PATH,
        survey_csv=SURVEY_PATH,
        target_label=target_selector.value,
        min_length=min_seq_slider.value,
        num_windows=num_windows_slider.value,
    )
    return (dataset,)


@app.cell
def _(dataset, mo, np, target_selector):
    if len(dataset) == 0:
        mo.md(
            f"**No samples available for {target_selector.value}. "
            "Adjust the minimum sequence length filter.**"
        )
        seq_lengths = []
    else:
        seq_lengths = [sample[0].shape[0] for sample in dataset.samples]
        summary = {
            "Users": len(seq_lengths),
            "Min length": min(seq_lengths),
            "Median length": int(np.median(seq_lengths)),
            "Max length": max(seq_lengths),
        }
        mo.vstack(
            [
                mo.md(f"### Dataset Summary ({target_selector.value})"),
                mo.ui.table(summary, selection=None),
            ]
        )
    return


@app.cell
def _(dataset, np, split_slider, train_test_split):

    user_ids = np.array(dataset.user_ids)
    train_ids, val_ids = train_test_split(
        user_ids,
        test_size=split_slider.value,
        random_state=42,
        shuffle=True,
    )

    id_to_index = {user_id: idx for idx, user_id in enumerate(dataset.user_ids)}
    train_indices = [id_to_index[user_id] for user_id in train_ids]
    val_indices = [id_to_index[user_id] for user_id in val_ids]
    return train_indices, val_indices


@app.cell
def _(
    BucketBatchSamplerSubset,
    DataLoader,
    batch_slider,
    collate_sequences,
    dataset,
    train_indices,
    val_indices,
):

    train_sampler = BucketBatchSamplerSubset(
        train_indices,
        dataset.lengths,
        batch_slider.value,
        shuffle=True,
    )

    val_sampler = BucketBatchSamplerSubset(
        val_indices,
        dataset.lengths,
        batch_slider.value,
        shuffle=False,
    )

    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_sequences,
    )
    val_loader = DataLoader(
        dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_sequences,
    )
    return train_loader, val_loader


@app.cell
def _(nn, np, pack_padded_sequence, torch):
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 10000):
            super().__init__()
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
            )
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x):
            length = x.size(1)
            return x + self.pe[:, :length]

    class SensorSequenceModel(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.1,
            model_type: str = "lstm",
        ):
            super().__init__()
            self.model_type = model_type

            if model_type == "lstm":
                self.encoder = nn.LSTM(
                    input_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.regressor = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 1),
                )
            else:
                nhead = 4 if hidden_dim % 4 == 0 else 1
                self.input_proj = nn.Linear(input_dim, hidden_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
                self.pos_encoder = PositionalEncoding(hidden_dim)
                self.regressor = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 1),
                )

        def forward(self, x, lengths):
            if self.model_type == "lstm":
                packed = pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                _, (hidden, _) = self.encoder(packed)
                features = hidden[-1]
            else:
                mask = self._padding_mask(lengths, x.size(1), x.device)
                x_proj = self.input_proj(x)
                x_proj = self.pos_encoder(x_proj)
                encoded = self.encoder(x_proj, src_key_padding_mask=mask)
                valid = (~mask).unsqueeze(-1).float()
                summed = (encoded * valid).sum(dim=1)
                counts = valid.sum(dim=1).clamp(min=1.0)
                features = summed / counts

            output = self.regressor(features).squeeze(-1)
            return output

        @staticmethod
        def _padding_mask(lengths, max_len, device):
            idx = torch.arange(max_len, device=device).unsqueeze(0)
            mask = idx >= lengths.unsqueeze(1)
            return mask
    return (SensorSequenceModel,)


@app.cell
def _(mean_absolute_error, mean_squared_error, np, r2_score, torch):
    def train_one_epoch(model, loader, criterion, optimizer, device):
        # lol
        model.train()
        total_loss = 0.0
        total_samples = 0
        for sequences, lengths, labels, _ in loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(sequences, lengths)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / max(total_samples, 1)

    def evaluate_model(model, loader, criterion, device):
        model.eval()
        total_loss = 0.0
        total_samples = 0
        preds_all, labels_all, ids_all = [], [], []

        with torch.no_grad():
            for sequences, lengths, labels, ids in loader:
                sequences = sequences.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)

                preds = model(sequences, lengths)
                loss = criterion(preds, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                preds_all.append(preds.cpu())
                labels_all.append(labels.cpu())
                ids_all.extend(ids)

        if total_samples == 0:
            return np.nan, np.array([]), np.array([]), [], {}

        preds_concat = torch.cat(preds_all).numpy()
        labels_concat = torch.cat(labels_all).numpy()

        metrics = {
            "MAE": mean_absolute_error(labels_concat, preds_concat),
            "RMSE": np.sqrt(mean_squared_error(labels_concat, preds_concat)),
        }

        if len(labels_concat) > 1:
            metrics["R2"] = float(r2_score(labels_concat, preds_concat))
        else:
            metrics["R2"] = float("nan")

        return (
            total_loss / total_samples,
            preds_concat,
            labels_concat,
            ids_all,
            metrics,
        )
    return evaluate_model, train_one_epoch


@app.cell
def _(
    SensorSequenceModel,
    copy,
    dropout_slider,
    epochs_slider,
    evaluate_model,
    hidden_slider,
    layers_slider,
    lr_slider,
    mo,
    model_selector,
    nn,
    np,
    target_selector,
    torch,
    train_button,
    train_loader,
    train_one_epoch,
    val_loader,
):
    if train_loader is None or val_loader is None or len(train_loader) == 0:
        mo.md("Adjust the data split / filters to create valid train and val sets.")
    
    if not train_button.value:
        mo.md("Set hyperparameters and click **Train model** to begin.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SensorSequenceModel(
        input_dim=train_loader.dataset.feature_dim,
        hidden_dim=hidden_slider.value,
        num_layers=layers_slider.value,
        dropout=dropout_slider.value,
        model_type=model_selector.value,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=10 ** lr_slider.value,
        weight_decay=1e-4,
    )
    criterion = nn.MSELoss()

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_snapshot = {}

    for epoch in range(1, epochs_slider.value + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"epoch {epoch} train loss: {train_loss}")
        val_loss, preds, labels, ids, metrics = evaluate_model(
            model, val_loader, criterion, device
        )
        print(f"epoch {epoch} val loss: {val_loss}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "MAE": metrics.get("MAE", np.nan),
                "RMSE": metrics.get("RMSE", np.nan),
                "R2": metrics.get("R2", np.nan),
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_snapshot = {
                "preds": preds,
                "labels": labels,
                "ids": ids,
                "metrics": metrics,
            }

    if best_state:
        model.load_state_dict(best_state)

    mo.md(
        f"Training complete for **{target_selector.value}** "
        f"({model_selector.value.upper()})"
    )
    return best_snapshot, history


@app.cell
def _(go, history, mo, pd):
    if not history:
        mo.md("Training history will appear after running the training cell.")
    
    history_df = pd.DataFrame(history)
    mo.vstack(
        [
            mo.md("### Training / Validation History"),
            mo.ui.table(history_df.round(4), selection=None),
        ]
    )

    fig = go.Figure()
    fig.add_scatter(
        x=history_df["epoch"],
        y=history_df["train_loss"],
        mode="lines+markers",
        name="Train Loss",
    )
    fig.add_scatter(
        x=history_df["epoch"],
        y=history_df["val_loss"],
        mode="lines+markers",
        name="Val Loss",
    )
    fig.update_layout(
        title="Loss Curves",
        xaxis_title="Epoch",
        yaxis_title="MSE Loss",
        height=400,
    )
    fig
    return


@app.cell
def _(best_snapshot, mo, np, pd):
    if not best_snapshot:
        mo.md("Run training to view validation predictions.")

    best_metrics = best_snapshot["metrics"]
    best_ids = best_snapshot["ids"]
    best_labels = best_snapshot["labels"]
    best_preds = best_snapshot["preds"]
    summary_train = {
        "Metric": list(best_metrics.keys()),
        "Value": [round(v, 4) if not np.isnan(v) else None for v in best_metrics.values()],
    }

    mo.vstack(
        [
            mo.md("### Validation Metrics"),
            mo.ui.table(summary_train, selection=None),
        ]
    )

    df_preds = pd.DataFrame(
        {"user_id": best_ids, "actual": best_labels, "predicted": best_preds}
    )
    mo.vstack(
        [
            mo.md("### Validation Predictions"),
            mo.ui.table(df_preds.round(3), selection=None),
        ]
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
