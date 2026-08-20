"""
Treina a LSTM de crescimento de gramínea (2 dias) com dados sintéticos.

Uso:
    python train_model.py
    python train_model.py --epochs 80 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from ml.config import (
    ARTIFACTS_DIR,
    MODEL_VERSION,
    SEED,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
)
from ml.metrics import format_metrics_table, regression_metrics
from ml.model_builder import build_lstm_model
from ml.preprocessing import (
    build_arrays,
    fit_scalers,
    inverse_transform_target,
    load_split,
    save_artifacts,
    transform_features,
    transform_target,
)


def set_seed(seed: int = SEED) -> None:
    keras.utils.set_random_seed(seed)


def evaluate_split(
    model,
    X_sequence_scaled: np.ndarray,
    X_future_scaled: np.ndarray,
    y_true_cm: np.ndarray,
    baseline_cm: np.ndarray,
    scaler_target,
) -> dict:
    scaled_pred = model.predict(
        {"sequence": X_sequence_scaled, "future": X_future_scaled},
        verbose=0,
    )
    y_pred_cm = inverse_transform_target(scaled_pred, scaler_target)
    y_true_flat = np.asarray(y_true_cm).flatten()
    return {
        "lstm": regression_metrics(y_true_flat, y_pred_cm),
        "baseline": regression_metrics(y_true_flat, baseline_cm),
    }


def train(epochs: int = 80, batch_size: int = 64) -> None:
    set_seed()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Carregando splits sintéticos (seed={SEED})")
    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VAL_PATH)
    test_df = load_split(TEST_PATH)

    train_arrays = build_arrays(train_df)
    val_arrays = build_arrays(val_df)
    test_arrays = build_arrays(test_df)

    print("Ajustando StandardScalers somente no treino...")
    scalers = fit_scalers(train_arrays["X_sequence"], train_arrays["X_future"], train_arrays["y"])

    X_train_seq, X_train_fut = transform_features(
        train_arrays["X_sequence"],
        train_arrays["X_future"],
        scalers["scaler_sequence"],
        scalers["scaler_future"],
    )
    X_val_seq, X_val_fut = transform_features(
        val_arrays["X_sequence"],
        val_arrays["X_future"],
        scalers["scaler_sequence"],
        scalers["scaler_future"],
    )
    X_test_seq, X_test_fut = transform_features(
        test_arrays["X_sequence"],
        test_arrays["X_future"],
        scalers["scaler_sequence"],
        scalers["scaler_future"],
    )

    y_train = transform_target(train_arrays["y"], scalers["scaler_target"])
    y_val = transform_target(val_arrays["y"], scalers["scaler_target"])

    print(
        f"Shapes | X_seq={X_train_seq.shape} | X_fut={X_train_fut.shape} | y={y_train.shape}"
    )

    model = build_lstm_model()
    callbacks = [
        EarlyStopping(
            monitor="val_mae",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    print(f"Treinando LSTM | epochs={epochs} | batch_size={batch_size}")
    history = model.fit(
        {"sequence": X_train_seq, "future": X_train_fut},
        y_train,
        validation_data=({"sequence": X_val_seq, "future": X_val_fut}, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    val_results = evaluate_split(
        model,
        X_val_seq,
        X_val_fut,
        val_arrays["y"],
        val_arrays["baseline"],
        scalers["scaler_target"],
    )
    test_results = evaluate_split(
        model,
        X_test_seq,
        X_test_fut,
        test_arrays["y"],
        test_arrays["baseline"],
        scalers["scaler_target"],
    )

    print("\nValidação")
    print(format_metrics_table(val_results["lstm"], val_results["baseline"]))
    print("\nTeste sintético")
    print(format_metrics_table(test_results["lstm"], test_results["baseline"]))

    lstm_mae = test_results["lstm"]["mae"]
    baseline_mae = test_results["baseline"]["mae"]
    if lstm_mae < baseline_mae:
        print(f"\nLSTM superou o baseline no teste (MAE {lstm_mae:.4f} < {baseline_mae:.4f} cm).")
    else:
        print(
            f"\nLSTM não superou o baseline no teste "
            f"(MAE {lstm_mae:.4f} >= {baseline_mae:.4f} cm)."
        )

    model_path = ARTIFACTS_DIR / "lstm_model.keras"
    model.save(model_path)

    metrics = {
        "validation": val_results,
        "test": test_results,
        "epochs_ran": int(len(history.history["loss"])),
        "best_val_mae_scaled": float(min(history.history["val_mae"])),
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    save_artifacts(
        scalers,
        extra_metadata={
            "model_version": MODEL_VERSION,
            "seed": SEED,
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "test_samples": int(len(test_df)),
        },
    )

    print(f"\nModelo salvo em: {model_path}")
    print(f"Scalers e metadata salvos em: {ARTIFACTS_DIR}")
    print("Proximo passo: python predict.py --input data/simulated_field/simulated_real_lstm_input.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina a LSTM de crescimento de gramínea")
    parser.add_argument("--epochs", type=int, default=80, help="Número máximo de epochs (padrão: 80)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
