from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.config import (
    ARTIFACTS_DIR,
    BASELINE_FORMULA,
    CONTEXT_DAYS,
    FUTURE_FEATURES,
    HORIZON_DAYS,
    LOOK_BACK,
    MODEL_VERSION,
    REQUIRED_INPUT_COLUMNS,
    SEQUENCE_FEATURE_ORDER,
    SEQUENCE_T1,
    SEQUENCE_T2,
    TARGET,
)


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_INPUT_COLUMNS + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em {path.name}: {missing}")
    return df


def build_arrays(df: pd.DataFrame, with_target: bool = True) -> dict[str, np.ndarray]:
    """Monta X_sequence (N, 2, 4), X_future (N, 3) e, se existir, y e baseline."""
    t1 = df[SEQUENCE_T1].to_numpy(dtype=np.float32)
    t2 = df[SEQUENCE_T2].to_numpy(dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "X_sequence": np.stack([t1, t2], axis=1),
        "X_future": df[FUTURE_FEATURES].to_numpy(dtype=np.float32),
        "baseline": (df["altura_t2_cm"] - df["altura_t1_cm"]).to_numpy(dtype=np.float32),
    }
    if with_target:
        if TARGET not in df.columns:
            raise ValueError(f"Coluna alvo '{TARGET}' não encontrada.")
        arrays["y"] = df[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)
    return arrays


def fit_scalers(X_sequence: np.ndarray, X_future: np.ndarray, y: np.ndarray) -> dict:
    """Fit somente no treino: scaler da sequência, do clima futuro e do alvo."""
    scaler_sequence = StandardScaler()
    scaler_future = StandardScaler()
    scaler_target = StandardScaler()

    n_features = X_sequence.shape[-1]
    scaler_sequence.fit(X_sequence.reshape(-1, n_features))
    scaler_future.fit(X_future)
    scaler_target.fit(y)

    return {
        "scaler_sequence": scaler_sequence,
        "scaler_future": scaler_future,
        "scaler_target": scaler_target,
    }


def transform_features(
    X_sequence: np.ndarray,
    X_future: np.ndarray,
    scaler_sequence: StandardScaler,
    scaler_future: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples, look_back, n_features = X_sequence.shape
    sequence_scaled = scaler_sequence.transform(X_sequence.reshape(-1, n_features))
    sequence_scaled = sequence_scaled.reshape(n_samples, look_back, n_features).astype(np.float32)
    future_scaled = scaler_future.transform(X_future).astype(np.float32)
    return sequence_scaled, future_scaled


def transform_target(y: np.ndarray, scaler_target: StandardScaler) -> np.ndarray:
    return scaler_target.transform(y).astype(np.float32)


def inverse_transform_target(values: np.ndarray, scaler_target: StandardScaler) -> np.ndarray:
    return scaler_target.inverse_transform(np.asarray(values).reshape(-1, 1)).flatten()


def save_artifacts(scalers: dict, extra_metadata: dict | None = None, output_dir: Path = ARTIFACTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers["scaler_sequence"], output_dir / "scaler_sequence.joblib")
    joblib.dump(scalers["scaler_future"], output_dir / "scaler_future.joblib")
    joblib.dump(scalers["scaler_target"], output_dir / "scaler_target.joblib")

    metadata = {
        "model_version": MODEL_VERSION,
        "look_back": LOOK_BACK,
        "context_days": CONTEXT_DAYS,
        "horizon_days": HORIZON_DAYS,
        "sequence_feature_order": SEQUENCE_FEATURE_ORDER,
        "sequence_t1": SEQUENCE_T1,
        "sequence_t2": SEQUENCE_T2,
        "future_features": FUTURE_FEATURES,
        "target": TARGET,
        "baseline": BASELINE_FORMULA,
        "scaler_type": "StandardScaler",
        "architecture": "LSTM(32) + concat future(3) + Dense(16) + Dense(1)",
        "feature_units": {
            "altura_cm": "cm",
            "temp_media_c": "C",
            "precipitacao_mm": "mm",
            "umidade_media_pct": "pct",
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    metadata = json.loads((artifacts_dir / "metadata.json").read_text(encoding="utf-8"))
    return {
        **metadata,
        "scaler_sequence": joblib.load(artifacts_dir / "scaler_sequence.joblib"),
        "scaler_future": joblib.load(artifacts_dir / "scaler_future.joblib"),
        "scaler_target": joblib.load(artifacts_dir / "scaler_target.joblib"),
    }
