"""
Gera um experimento de campo simulado (D0 → D2 → D4) para testar a previsão.

D0→D2 e D2→D4 representam 2 dias. O crescimento real de D4 é holdout:
a LSTM em D2 vê apenas contexto + previsão climática.

Uso:
    python simulate_field_data.py

Saídas:
    data/simulated_field/simulated_field_collection.csv
    data/simulated_field/simulated_real_lstm_input.csv
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ml.config import (
    CONTEXT_DAYS,
    FIELD_SEED,
    HORIZON_DAYS,
    PAST_GROWTH,
    SIMULATED_FIELD_COLLECTION_PATH,
    SIMULATED_REAL_INPUT_PATH,
)
from ml.growth_simulator import (
    HEIGHT_CLIP_CM,
    HEIGHT_MIN_CM,
    HEIGHT_STD_CM,
    aggregate_weather,
    forecast_from_actual,
    growth_2d,
)

N_POINTS = 5
N_SITES = 10

D0_BASE = datetime(2026, 8, 7, 9, 20)
D2_BASE = D0_BASE + timedelta(days=CONTEXT_DAYS)
D4_BASE = D0_BASE + timedelta(days=CONTEXT_DAYS + HORIZON_DAYS)

# Alturas de gramado alto, na mesma faixa da coleta REAL001 (53–80 cm, média 62,25).
SITES = [
    {"sample_id": "REAL001", "altura0": 62.25, "clima": "frio"},
    {"sample_id": "REAL002", "altura0": 54.80, "clima": "chuva"},
    {"sample_id": "REAL003", "altura0": 71.40, "clima": "medio"},
    {"sample_id": "REAL004", "altura0": 48.50, "clima": "quente"},
    {"sample_id": "REAL005", "altura0": 78.20, "clima": "umido"},
    {"sample_id": "REAL006", "altura0": 58.00, "clima": "frio"},
    {"sample_id": "REAL007", "altura0": 66.70, "clima": "seco"},
    {"sample_id": "REAL008", "altura0": 52.30, "clima": "chuva"},
    {"sample_id": "REAL009", "altura0": 80.00, "clima": "medio"},
    {"sample_id": "REAL010", "altura0": 45.60, "clima": "quente"},
]

CLIMATE_PROFILES = {
    "seco": {"temp": 26.5, "rain_p": 0.08, "rain_scale": 4.0, "humidity": 58.0},
    "chuva": {"temp": 22.8, "rain_p": 0.85, "rain_scale": 9.0, "humidity": 82.0},
    "medio": {"temp": 24.0, "rain_p": 0.30, "rain_scale": 7.0, "humidity": 70.0},
    "quente": {"temp": 29.2, "rain_p": 0.18, "rain_scale": 6.0, "humidity": 62.0},
    "umido": {"temp": 23.4, "rain_p": 0.55, "rain_scale": 8.0, "humidity": 86.0},
    "frio": {"temp": 18.6, "rain_p": 0.22, "rain_scale": 5.0, "humidity": 74.0},
}


def _round(value: float, digits: int = 3) -> float:
    return float(round(value, digits))


def sample_weather(profile: str, rng: np.random.Generator) -> dict[str, float]:
    cfg = CLIMATE_PROFILES[profile]
    temp = float(np.clip(rng.normal(cfg["temp"], 1.6), 12.0, 35.0))
    rain = float(rng.exponential(cfg["rain_scale"])) if rng.random() < cfg["rain_p"] else 0.0
    rain = float(np.clip(rain, 0.0, 35.0))
    humidity = float(np.clip(rng.normal(cfg["humidity"], 5.5), 35.0, 98.0))
    return {
        "temp": _round(temp, 2),
        "rain": _round(rain, 2),
        "humidity": _round(humidity, 2),
    }


def sample_point_offsets(rng: np.random.Generator) -> np.ndarray:
    """Offsets fixos dos 5 pontos (mesma estaca em D0, D2 e D4)."""
    spatial_std = min(7.0, max(4.0, HEIGHT_STD_CM * 0.75))
    return rng.normal(0.0, spatial_std, size=N_POINTS)


def measure_points(
    mean_height: float,
    offsets: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[float], float]:
    measurement_noise = rng.normal(0.0, 0.12, size=N_POINTS)
    points = mean_height + offsets + measurement_noise
    points = np.clip(points, HEIGHT_MIN_CM, HEIGHT_CLIP_CM)
    points = [_round(float(p), 3) for p in points]
    return points, _round(float(np.mean(points)), 3)


def build_experiment(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    collection_rows: list[dict] = []
    input_rows: list[dict] = []

    for index, site in enumerate(SITES[:N_SITES]):
        offset = timedelta(minutes=int(index * 7))
        d0_time = D0_BASE + offset
        d2_time = D2_BASE + offset
        d4_time = D4_BASE + offset
        profile = site["clima"]

        weather_d0 = sample_weather(profile, rng)
        weather_d1 = sample_weather(profile, rng)
        weather_d2 = sample_weather(profile, rng)
        weather_d3 = sample_weather(profile, rng)
        weather_d4 = sample_weather(profile, rng)

        context_weather = aggregate_weather(weather_d0, weather_d1)
        actual_future = aggregate_weather(weather_d3, weather_d4)
        forecast_future = forecast_from_actual(actual_future, rng)

        offsets = sample_point_offsets(rng)
        true_d0 = float(site["altura0"])
        points_d0, altura_d0 = measure_points(true_d0, offsets, rng)

        growth_d0_d2 = growth_2d(true_d0, context_weather, rng, days=CONTEXT_DAYS)
        true_d2 = _round(true_d0 + growth_d0_d2, 3)
        points_d2, altura_d2 = measure_points(true_d2, offsets, rng)

        growth_d2_d4 = growth_2d(true_d2, actual_future, rng, days=HORIZON_DAYS)
        true_d4 = _round(true_d2 + growth_d2_d4, 3)
        points_d4, altura_d4 = measure_points(true_d4, offsets, rng)
        crescimento_real = _round(altura_d4 - altura_d2, 3)

        phases = [
            ("D0_CONTEXTO", d0_time, points_d0, altura_d0, weather_d0, "Observado (API simulada)", "Primeira medição da área fixa. Janela de contexto: 2 dias."),
            (
                "D2_PREVISAO",
                d2_time,
                points_d2,
                altura_d2,
                weather_d2,
                "Observado (API simulada)",
                (
                    f"Momento da previsão. Forecast D2->D4: "
                    f"temp={forecast_future['temp']}C, "
                    f"chuva={forecast_future['rain']}mm, "
                    f"umidade={forecast_future['humidity']}%."
                ),
            ),
            (
                "D4_VALIDACAO",
                d4_time,
                points_d4,
                altura_d4,
                weather_d4,
                "Observado (API simulada)",
                "Usado somente para validar. Nao entra como feature da LSTM.",
            ),
        ]

        for fase, quando, pontos, media, clima, fonte, obs in phases:
            collection_rows.append(
                {
                    "experiment_id": site["sample_id"],
                    "data_hora": quando.strftime("%Y-%m-%d %H:%M:%S"),
                    "fase": fase,
                    "ponto_1_cm": pontos[0],
                    "ponto_2_cm": pontos[1],
                    "ponto_3_cm": pontos[2],
                    "ponto_4_cm": pontos[3],
                    "ponto_5_cm": pontos[4],
                    "altura_media_cm": media,
                    "temp_media_c": clima["temp"],
                    "precipitacao_mm": clima["rain"],
                    "umidade_media_pct": clima["humidity"],
                    "fonte_clima": fonte,
                    "observacoes": obs,
                }
            )

        input_rows.append(
            {
                "sample_id": site["sample_id"],
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
                "crescimento_futuro_2d_cm": "",
                "data_previsao": d2_time.strftime("%Y-%m-%d"),
                "altura_real_d4_cm": altura_d4,
                "crescimento_real_2d_cm": crescimento_real,
                "crescimento_previsto_2d_cm": "",
                "erro_absoluto_cm": "",
            }
        )

    return pd.DataFrame(collection_rows), pd.DataFrame(input_rows)


def main() -> None:
    rng = np.random.default_rng(FIELD_SEED)
    collection_df, input_df = build_experiment(rng)

    SIMULATED_FIELD_COLLECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    collection_df.to_csv(SIMULATED_FIELD_COLLECTION_PATH, index=False)
    input_df.to_csv(SIMULATED_REAL_INPUT_PATH, index=False)

    print(f"Coleta de campo simulada: {SIMULATED_FIELD_COLLECTION_PATH}")
    print(f"Entrada LSTM simulada:    {SIMULATED_REAL_INPUT_PATH}")
    print("\nCrescimento real D2->D4 (holdout, nao entra no modelo):")
    print(
        input_df[["sample_id", "altura_t2_cm", "altura_real_d4_cm", "crescimento_real_2d_cm"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
