"""
Regenera o dataset sintético com a escala temporal correta:

    D0 --2 dias--> D2 --2 dias--> D4

Uso:
    python generate_synthetic_data.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import (
    CONTEXT_DAYS,
    FULL_SYNTHETIC_PATH,
    HORIZON_DAYS,
    PAST_GROWTH,
    SEED,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
)
from ml.growth_simulator import (
    HEIGHT_CLIP_CM,
    HEIGHT_MAX_CM,
    HEIGHT_MEAN_CM,
    HEIGHT_MIN_CM,
    HEIGHT_STD_CM,
    aggregate_weather,
    forecast_from_actual,
    growth_2d,
    sample_climate_day,
)

N_SAMPLES = 10_000
TRAIN_N = 7_000
VAL_N = 1_500
TEST_N = 1_500


def _round(value: float, digits: int = 3) -> float:
    return float(round(value, digits))


def build_sample(index: int, rng: np.random.Generator) -> dict:
    weather_d0 = sample_climate_day(rng)
    weather_d1 = sample_climate_day(rng)
    weather_d2 = sample_climate_day(rng)
    weather_d3 = sample_climate_day(rng)
    weather_d4 = sample_climate_day(rng)

    context_weather = aggregate_weather(weather_d0, weather_d1)
    actual_future = aggregate_weather(weather_d3, weather_d4)
    forecast_future = forecast_from_actual(actual_future, rng)

    altura_d0 = _round(
        float(np.clip(rng.normal(HEIGHT_MEAN_CM, HEIGHT_STD_CM), HEIGHT_MIN_CM, HEIGHT_MAX_CM)),
        3,
    )
    crescimento_passado = _round(growth_2d(altura_d0, context_weather, rng, days=CONTEXT_DAYS), 3)
    altura_d2 = _round(float(np.clip(altura_d0 + crescimento_passado, HEIGHT_MIN_CM, HEIGHT_CLIP_CM)), 3)
    crescimento_futuro = _round(growth_2d(altura_d2, actual_future, rng, days=HORIZON_DAYS), 3)

    return {
        "sample_id": f"S{index:05d}",
        "altura_t1_cm": altura_d0,
        "temp_t1_c": weather_d0["temp"],
        "precipitacao_t1_mm": weather_d0["rain"],
        "umidade_t1_pct": weather_d0["humidity"],
        "altura_t2_cm": altura_d2,
        "temp_t2_c": weather_d2["temp"],
        "precipitacao_t2_mm": weather_d2["rain"],
        "umidade_t2_pct": weather_d2["humidity"],
        "temp_futuro_2d_c": forecast_future["temp"],
        "precipitacao_futuro_2d_mm": forecast_future["rain"],
        "umidade_futuro_2d_pct": forecast_future["humidity"],
        PAST_GROWTH: _round(altura_d2 - altura_d0, 3),
        TARGET: crescimento_futuro,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = [build_sample(index + 1, rng) for index in range(N_SAMPLES)]
    full = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    train_df = full.iloc[:TRAIN_N].copy()
    val_df = full.iloc[TRAIN_N : TRAIN_N + VAL_N].copy()
    test_df = full.iloc[TRAIN_N + VAL_N : TRAIN_N + VAL_N + TEST_N].copy()

    FULL_SYNTHETIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(FULL_SYNTHETIC_PATH, index=False)
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"full={len(full)} train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(full[["altura_t1_cm", "altura_t2_cm", PAST_GROWTH, TARGET]].describe().round(3).to_string())
    print(f"\nSalvo em: {FULL_SYNTHETIC_PATH.parent}")


if __name__ == "__main__":
    main()
