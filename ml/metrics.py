from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def format_metrics_table(lstm: dict[str, float], baseline: dict[str, float]) -> str:
    header = f"{'':<10} {'MAE (cm)':>12} {'RMSE (cm)':>12} {'R²':>10}"
    lstm_row = f"{'LSTM':<10} {lstm['mae']:>12.4f} {lstm['rmse']:>12.4f} {lstm['r2']:>10.4f}"
    base_row = f"{'Baseline':<10} {baseline['mae']:>12.4f} {baseline['rmse']:>12.4f} {baseline['r2']:>10.4f}"
    return "\n".join([header, lstm_row, base_row])
