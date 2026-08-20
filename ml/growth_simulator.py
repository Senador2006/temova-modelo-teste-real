from __future__ import annotations

import numpy as np

from ml.config import CONTEXT_DAYS


def _round(value: float, digits: int = 3) -> float:
    return float(round(value, digits))


def aggregate_weather(day_a: dict[str, float], day_b: dict[str, float]) -> dict[str, float]:
    """Temp/umidade = média da janela de 2 dias; precipitação = acumulado."""
    return {
        "temp": _round((day_a["temp"] + day_b["temp"]) / 2.0, 2),
        "rain": _round(day_a["rain"] + day_b["rain"], 2),
        "humidity": _round((day_a["humidity"] + day_b["humidity"]) / 2.0, 2),
    }


def forecast_from_actual(actual: dict[str, float], rng: np.random.Generator) -> dict[str, float]:
    """Previsão disponível em D2 — próxima da realidade, com erro de forecast."""
    return {
        "temp": _round(float(np.clip(actual["temp"] + rng.normal(0, 1.1), 12.0, 35.0)), 2),
        "rain": _round(float(np.clip(actual["rain"] + rng.normal(0, 2.4), 0.0, 45.0)), 2),
        "humidity": _round(float(np.clip(actual["humidity"] + rng.normal(0, 3.8), 35.0, 98.0)), 2),
    }


# Gramado alto / maduro, alinhado à 1ª coleta real (REAL001):
# D0 = 62,25 cm (pontos 53–80 cm), D2 = 62,37 cm.
HEIGHT_MEAN_CM = 62.0
HEIGHT_STD_CM = 8.5
HEIGHT_MIN_CM = 40.0
HEIGHT_MAX_CM = 90.0
HEIGHT_CLIP_CM = 110.0

# Taxa diária de sward maduro: REAL001 cresceu ~0,12 cm em ~1 dia
# (clima fresco ~21 °C). Gramínea baixa recém-podada crescia bem mais.
DAILY_GROWTH_MIN_CM = 0.02
DAILY_GROWTH_MAX_CM = 0.40
DAILY_GROWTH_BASE_CM = 0.12
TEMP_COEF = 0.010
RAIN_COEF = 0.010
HUMIDITY_COEF = 0.0025
HEIGHT_COEF = 0.0035


def growth_cm(
    altura: float,
    temp: float,
    rain: float,
    humidity: float,
    days: int,
    rng: np.random.Generator,
    rain_accumulated: bool = False,
) -> float:
    """Crescimento simplificado para a janela informada (D0→D2 ou D2→D4)."""
    rain_daily = rain / days if rain_accumulated and days else rain
    daily = (
        DAILY_GROWTH_BASE_CM
        + TEMP_COEF * (temp - 24.0)
        + RAIN_COEF * rain_daily
        + HUMIDITY_COEF * (humidity - 70.0)
        - HEIGHT_COEF * (altura - HEIGHT_MEAN_CM)
    )
    daily = float(np.clip(daily, DAILY_GROWTH_MIN_CM, DAILY_GROWTH_MAX_CM))
    noise = float(rng.normal(0.0, 0.02 * np.sqrt(days)))
    window_min = max(0.0, DAILY_GROWTH_MIN_CM * days - 0.03)
    window_max = DAILY_GROWTH_MAX_CM * days + 0.08
    return float(np.clip(daily * days + noise, window_min, window_max))


def growth_2d(
    altura: float,
    weather: dict[str, float],
    rng: np.random.Generator,
    days: int = CONTEXT_DAYS,
) -> float:
    return growth_cm(
        altura=altura,
        temp=weather["temp"],
        rain=weather["rain"],
        humidity=weather["humidity"],
        days=days,
        rng=rng,
        rain_accumulated=True,
    )


def sample_climate_day(rng: np.random.Generator) -> dict[str, float]:
    """Clima diário coerente com a faixa do Rodoanel / SP."""
    season = rng.choice(["verao", "inverno", "transicao"], p=[0.25, 0.25, 0.50])
    if season == "verao":
        temp_mu, rain_p, rain_scale, hum_mu = 25.5, 0.42, 10.0, 74.0
    elif season == "inverno":
        temp_mu, rain_p, rain_scale, hum_mu = 18.5, 0.16, 5.0, 66.0
    else:
        temp_mu, rain_p, rain_scale, hum_mu = 23.5, 0.28, 7.5, 70.0

    temp = float(np.clip(rng.normal(temp_mu, 2.4), 12.0, 35.0))
    rain = float(rng.exponential(rain_scale)) if rng.random() < rain_p else 0.0
    rain = float(np.clip(rain, 0.0, 35.0))
    humidity = float(np.clip(rng.normal(hum_mu, 8.0), 35.0, 98.0))
    return {
        "temp": _round(temp, 2),
        "rain": _round(rain, 2),
        "humidity": _round(humidity, 2),
    }
