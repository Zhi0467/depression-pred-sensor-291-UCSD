"""
Unit tests for baseline linear model components.

Tests:
- Feature aggregation functions
- LinearRegressor model
- Training with regularization
- Metric computation
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import numpy as np
import pytest


# ============================================================================
# Re-implement functions here for testing (avoid notebook import issues)
# ============================================================================

class LinearRegressor(nn.Module):
    """Simple linear regression model for predicting clinical scores."""

    def __init__(self, input_dim: int, output_dim: int = 1):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_weights(self):
        return self.linear.weight.data, self.linear.bias.data


def aggregate_sequences(seq_tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Aggregate variable-length sequences to fixed-length features."""
    batch_size = seq_tensor.shape[0]
    num_features = seq_tensor.shape[2]
    aggregated = []

    for i in range(batch_size):
        length = int(lengths[i].item())
        if length == 0:
            stats = torch.zeros(num_features * 4)
        else:
            seq = seq_tensor[i, :length, :]
            mean_vals = seq.mean(dim=0)
            std_vals = seq.std(dim=0) if length > 1 else torch.zeros(num_features)
            min_vals = seq.min(dim=0)[0]
            max_vals = seq.max(dim=0)[0]
            stats = torch.cat([mean_vals, std_vals, min_vals, max_vals])
        aggregated.append(stats)

    return torch.stack(aggregated)


def compute_regularization(model: nn.Module, reg_type: str, alpha: float) -> torch.Tensor:
    """Compute regularization penalty."""
    if reg_type == 'none' or alpha == 0:
        return torch.tensor(0.0)

    reg_loss = torch.tensor(0.0)
    for param in model.parameters():
        if reg_type == 'l1':
            reg_loss = reg_loss + torch.sum(torch.abs(param))
        elif reg_type == 'l2':
            reg_loss = reg_loss + torch.sum(param ** 2)

    return alpha * reg_loss


def compute_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """Compute regression evaluation metrics."""
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


# ============================================================================
# Tests
# ============================================================================

class TestLinearRegressor:
    """Tests for LinearRegressor model."""

    def test_initialization(self):
        """Test model initialization with different dimensions."""
        model = LinearRegressor(input_dim=10, output_dim=1)
        assert model.linear.in_features == 10
        assert model.linear.out_features == 1

        model = LinearRegressor(input_dim=24, output_dim=3)
        assert model.linear.in_features == 24
        assert model.linear.out_features == 3

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = LinearRegressor(input_dim=10, output_dim=1)
        x = torch.randn(5, 10)  # Batch of 5
        y = model(x)
        assert y.shape == (5, 1)

        model = LinearRegressor(input_dim=10, output_dim=3)
        y = model(x)
        assert y.shape == (5, 3)

    def test_get_weights(self):
        """Test weight retrieval."""
        model = LinearRegressor(input_dim=10, output_dim=1)
        weights, bias = model.get_weights()
        assert weights.shape == (1, 10)
        assert bias.shape == (1,)

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        model = LinearRegressor(input_dim=5, output_dim=1)
        x = torch.randn(3, 5)
        y_true = torch.randn(3, 1)

        y_pred = model(x)
        loss = nn.MSELoss()(y_pred, y_true)
        loss.backward()

        # Check gradients exist
        assert model.linear.weight.grad is not None
        assert model.linear.bias.grad is not None


class TestAggregateSequences:
    """Tests for sequence aggregation function."""

    def test_basic_aggregation(self):
        """Test basic sequence aggregation."""
        # 2 samples, max_len=5, 3 features
        seq = torch.tensor([
            [[1.0, 2.0, 3.0],
             [2.0, 3.0, 4.0],
             [3.0, 4.0, 5.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]]
        ])
        lengths = torch.tensor([3, 2])

        result = aggregate_sequences(seq, lengths)

        # Output should be (2, 3*4=12)
        assert result.shape == (2, 12)

        # Check first sample mean (features 0-2)
        expected_mean = torch.tensor([2.0, 3.0, 4.0])
        assert torch.allclose(result[0, :3], expected_mean)

        # Check second sample (all 1s, so std=0)
        assert torch.allclose(result[1, 3:6], torch.zeros(3))  # std = 0

    def test_single_element_sequence(self):
        """Test aggregation with single element sequence."""
        seq = torch.tensor([[[1.0, 2.0]]])
        lengths = torch.tensor([1])

        result = aggregate_sequences(seq, lengths)

        # std should be 0 for single element
        assert result[0, 2] == 0.0  # std of first feature
        assert result[0, 3] == 0.0  # std of second feature

    def test_empty_sequence(self):
        """Test aggregation with empty sequence."""
        seq = torch.zeros(1, 5, 3)
        lengths = torch.tensor([0])

        result = aggregate_sequences(seq, lengths)

        # All zeros for empty sequence
        assert torch.allclose(result, torch.zeros(1, 12))

    def test_output_order(self):
        """Test that output is [mean, std, min, max]."""
        seq = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
        lengths = torch.tensor([5])

        result = aggregate_sequences(seq, lengths)

        # 1 feature * 4 stats = 4
        assert result.shape == (1, 4)

        # mean=3, std~1.58, min=1, max=5
        assert abs(result[0, 0].item() - 3.0) < 0.01  # mean
        assert result[0, 2].item() == 1.0  # min
        assert result[0, 3].item() == 5.0  # max


class TestRegularization:
    """Tests for regularization computation."""

    def test_no_regularization(self):
        """Test that no regularization returns 0."""
        model = LinearRegressor(5, 1)
        reg = compute_regularization(model, 'none', 0.1)
        assert reg.item() == 0.0

        reg = compute_regularization(model, 'l2', 0.0)
        assert reg.item() == 0.0

    def test_l2_regularization(self):
        """Test L2 (Ridge) regularization."""
        model = LinearRegressor(2, 1)
        # Set known weights
        model.linear.weight.data = torch.tensor([[1.0, 2.0]])
        model.linear.bias.data = torch.tensor([0.5])

        # L2 = alpha * (1^2 + 2^2 + 0.5^2) = alpha * 5.25
        reg = compute_regularization(model, 'l2', 1.0)
        assert abs(reg.item() - 5.25) < 0.01

        reg = compute_regularization(model, 'l2', 0.1)
        assert abs(reg.item() - 0.525) < 0.01

    def test_l1_regularization(self):
        """Test L1 (Lasso) regularization."""
        model = LinearRegressor(2, 1)
        model.linear.weight.data = torch.tensor([[1.0, -2.0]])
        model.linear.bias.data = torch.tensor([0.5])

        # L1 = alpha * (|1| + |-2| + |0.5|) = alpha * 3.5
        reg = compute_regularization(model, 'l1', 1.0)
        assert abs(reg.item() - 3.5) < 0.01


class TestMetrics:
    """Tests for metric computation."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = compute_metrics(predictions, actuals)

        assert metrics['MAE'] == 0.0
        assert metrics['RMSE'] == 0.0
        assert metrics['R2'] == 1.0

    def test_mae_calculation(self):
        """Test MAE calculation."""
        actuals = np.array([1.0, 2.0, 3.0])
        predictions = np.array([2.0, 2.0, 1.0])

        # MAE = (1 + 0 + 2) / 3 = 1.0
        metrics = compute_metrics(predictions, actuals)
        assert abs(metrics['MAE'] - 1.0) < 0.01

    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        actuals = np.array([1.0, 2.0, 3.0])
        predictions = np.array([2.0, 2.0, 1.0])

        # MSE = (1 + 0 + 4) / 3 = 5/3
        # RMSE = sqrt(5/3) ~ 1.29
        metrics = compute_metrics(predictions, actuals)
        assert abs(metrics['RMSE'] - np.sqrt(5/3)) < 0.01

    def test_r2_calculation(self):
        """Test R² calculation."""
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

        metrics = compute_metrics(predictions, actuals)

        # R² should be close to 1 for good predictions
        assert metrics['R2'] > 0.99

    def test_negative_r2(self):
        """Test that R² can be negative for poor predictions."""
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Reversed

        metrics = compute_metrics(predictions, actuals)

        # R² should be negative
        assert metrics['R2'] < 0


class TestTrainingIntegration:
    """Integration tests for training pipeline."""

    def test_simple_linear_fit(self):
        """Test that model can fit simple linear relationship."""
        # Generate simple linear data: y = 2*x + 1
        torch.manual_seed(42)
        X = torch.randn(20, 1)
        y = 2 * X + 1 + 0.1 * torch.randn(20, 1)  # Small noise

        model = LinearRegressor(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        # Train
        for _ in range(500):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        # Check learned parameters
        weights, bias = model.get_weights()
        assert abs(weights[0, 0].item() - 2.0) < 0.3
        assert abs(bias[0].item() - 1.0) < 0.3

    def test_multivariate_fit(self):
        """Test fitting with multiple features."""
        torch.manual_seed(42)
        X = torch.randn(30, 3)
        # y = 1*x1 + 2*x2 + 3*x3
        true_weights = torch.tensor([[1.0, 2.0, 3.0]])
        y = X @ true_weights.T

        model = LinearRegressor(3, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        # Train
        for _ in range(1000):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        # Final loss should be very small
        with torch.no_grad():
            final_loss = criterion(model(X), y)
        assert final_loss.item() < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
