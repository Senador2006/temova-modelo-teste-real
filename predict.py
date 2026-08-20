"""
Gera previsão D2→D4 a partir de um CSV no formato do template de entrada.

Uso:
    python predict.py
    python predict.py --input data/simulated_field/simulated_real_lstm_input.csv
    python predict.py --input data/templates/real_lstm_input_template.csv --output results/predicoes_reais.csv
"""
from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("KERAS_BACKEND", "torch")

from keras.models import load_model

from ml.config import (
    ARTIFACTS_DIR,
    REQUIRED_INPUT_COLUMNS,
    RESULTS_DIR,
    SIMULATED_REAL_INPUT_PATH,
)
from ml.metrics import format_metrics_table, regression_metrics
from ml.preprocessing import (
    build_arrays,
    inverse_transform_target,
    load_artifacts,
    transform_features,
)

_model_cache: dict[str, object] = {}


def _load_model_cached(artifacts_dir: Path):
    key = str(artifacts_dir.resolve())
    cached = _model_cache.get(key)
    if cached is not None:
        return cached
    artifacts = load_artifacts(artifacts_dir)
    model = load_model(artifacts_dir / "lstm_model.keras")
    bundle = (model, artifacts)
    _model_cache[key] = bundle
    return bundle


def _series_or_empty(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def attach_real_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche crescimento real de D4 só para avaliação — não entra na LSTM."""
    out = df.copy()
    altura_d4 = _series_or_empty(out, "altura_real_d4_cm")
    crescimento_real = _series_or_empty(out, "crescimento_real_2d_cm")

    missing_real = crescimento_real.isna() & altura_d4.notna()
    if missing_real.any():
        crescimento_real = crescimento_real.copy()
        crescimento_real.loc[missing_real] = (
            altura_d4.loc[missing_real] - out.loc[missing_real, "altura_t2_cm"]
        )

    out["crescimento_real_2d_cm"] = crescimento_real.round(4)
    if out["crescimento_real_2d_cm"].notna().any():
        out["erro_absoluto_cm"] = (
            (out["crescimento_previsto_2d_cm"] - out["crescimento_real_2d_cm"]).abs().round(4)
        )
        out["erro_absoluto_baseline_cm"] = (
            (out["baseline_previsto_2d_cm"] - out["crescimento_real_2d_cm"]).abs().round(4)
        )
    return out


def predict_dataframe(df: pd.DataFrame, artifacts_dir: Path = ARTIFACTS_DIR) -> pd.DataFrame:
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas de entrada ausentes: {missing}")

    ready = df.dropna(subset=REQUIRED_INPUT_COLUMNS).copy()
    if ready.empty:
        raise ValueError("Nenhuma linha com as features de entrada preenchidas.")

    model, artifacts = _load_model_cached(artifacts_dir)
    arrays = build_arrays(ready, with_target=False)
    X_seq, X_fut = transform_features(
        arrays["X_sequence"],
        arrays["X_future"],
        artifacts["scaler_sequence"],
        artifacts["scaler_future"],
    )
    scaled_pred = model.predict({"sequence": X_seq, "future": X_fut}, verbose=0)
    ready["crescimento_previsto_2d_cm"] = inverse_transform_target(
        scaled_pred,
        artifacts["scaler_target"],
    ).round(4)
    ready["baseline_previsto_2d_cm"] = arrays["baseline"].round(4)
    if "data_previsao" not in ready.columns:
        ready["data_previsao"] = date.today().isoformat()
    else:
        ready["data_previsao"] = ready["data_previsao"].fillna(date.today().isoformat())
    ready["model_version"] = artifacts.get("model_version", "unknown")
    return attach_real_growth(ready)


def print_evaluation(predicted: pd.DataFrame) -> None:
    cols = [
        "sample_id",
        "crescimento_previsto_2d_cm",
        "baseline_previsto_2d_cm",
        "crescimento_real_2d_cm",
        "erro_absoluto_cm",
    ]
    available = [col for col in cols if col in predicted.columns]
    print(predicted[available].to_string(index=False))

    real = _series_or_empty(predicted, "crescimento_real_2d_cm").dropna()
    if real.empty:
        return

    aligned = predicted.loc[real.index]
    lstm = regression_metrics(aligned["crescimento_real_2d_cm"], aligned["crescimento_previsto_2d_cm"])
    baseline = regression_metrics(aligned["crescimento_real_2d_cm"], aligned["baseline_previsto_2d_cm"])
    print("\nComparação com crescimento real de D4")
    print(format_metrics_table(lstm, baseline))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prevê crescimento de gramínea nos próximos 2 dias")
    parser.add_argument("--input", type=Path, default=SIMULATED_REAL_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "predicoes.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    predicted = predict_dataframe(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predicted.to_csv(args.output, index=False)

    print_evaluation(predicted)
    print(f"\nSalvo em: {args.output}")


if __name__ == "__main__":
    main()
