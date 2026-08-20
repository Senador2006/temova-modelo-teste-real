"""API de inferência do modelo de campo (LSTM 2.3.0, horizonte D2→D4)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

os.environ.setdefault("KERAS_BACKEND", "torch")

from ml.config import (  # noqa: E402
    ARTIFACTS_DIR,
    LOOK_BACK,
    MODEL_VERSION,
    REQUIRED_INPUT_COLUMNS,
    SIMULATED_REAL_INPUT_PATH,
)
from predict import predict_dataframe  # noqa: E402

app = FastAPI(
    title="API MLOps — Modelo de campo (gramínea D2→D4)",
    description=(
        "Serviço de inferência do LSTM 2.3.0. Em D2 prevê o crescimento até D4. "
        "A altura real de D4 é opcional e só entra no cálculo de erro."
    ),
    version=MODEL_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CampoRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    sample_id: str | None = None
    altura_t1_cm: float
    temp_t1_c: float
    precipitacao_t1_mm: float
    umidade_t1_pct: float
    altura_t2_cm: float
    temp_t2_c: float
    precipitacao_t2_mm: float
    umidade_t2_pct: float
    temp_futuro_2d_c: float
    precipitacao_futuro_2d_mm: float
    umidade_futuro_2d_pct: float
    data_previsao: str | None = None
    altura_real_d4_cm: float | None = None


class CampoPredictRequest(BaseModel):
    records: list[CampoRecord] = Field(min_length=1)


@app.on_event("startup")
def warmup_model() -> None:
    if not (ARTIFACTS_DIR / "lstm_model.keras").exists():
        raise RuntimeError(
            f"Artefatos não encontrados em {ARTIFACTS_DIR}. "
            "Treine com train_model.py e versionar artifacts/."
        )
    dummy = pd.DataFrame(
        [
            {
                "altura_t1_cm": 62.0,
                "temp_t1_c": 21.0,
                "precipitacao_t1_mm": 0.0,
                "umidade_t1_pct": 80.0,
                "altura_t2_cm": 62.2,
                "temp_t2_c": 21.0,
                "precipitacao_t2_mm": 0.0,
                "umidade_t2_pct": 80.0,
                "temp_futuro_2d_c": 22.0,
                "precipitacao_futuro_2d_mm": 0.0,
                "umidade_futuro_2d_pct": 70.0,
            }
        ]
    )
    predict_dataframe(dummy)


@app.get("/health", tags=["MLOps"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "look_back": LOOK_BACK,
        "horizon": "D2→D4 (2 dias)",
    }


@app.get("/model/info", tags=["MLOps"])
def model_info() -> dict[str, Any]:
    return {
        "modelo": "LSTM(32) + clima futuro",
        "versao": MODEL_VERSION,
        "alvo": "crescimento_futuro_2d_cm",
        "contexto": "D0 e D2 (2 timesteps × 4 features)",
        "clima_futuro": "temp, chuva e umidade previstos para os próximos 2 dias",
        "features_obrigatorias": REQUIRED_INPUT_COLUMNS,
        "anti_leakage": "altura de D4 nunca entra como feature",
    }


@app.post("/predict", tags=["Inferência"])
def predict(request: CampoPredictRequest) -> dict[str, Any]:
    df = pd.DataFrame([row.model_dump() for row in request.records])
    try:
        predicted = predict_dataframe(df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {exc}") from exc

    records = predicted.to_dict(orient="records")
    for row in records:
        for key, value in list(row.items()):
            if pd.isna(value):
                row[key] = None
            elif hasattr(value, "item"):
                row[key] = value.item()
    return {
        "model_version": MODEL_VERSION,
        "n": len(records),
        "records": records,
    }


@app.get("/sample/{sample_id}", tags=["Inferência"])
def get_sample(sample_id: str = "REAL001") -> dict[str, Any]:
    path = Path(SIMULATED_REAL_INPUT_PATH)
    if not path.exists():
        raise HTTPException(status_code=404, detail="CSV de amostra não encontrado.")
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise HTTPException(status_code=500, detail="CSV sem coluna sample_id.")
    match = df[df["sample_id"].astype(str) == sample_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Amostra {sample_id} não encontrada.")
    row = match.iloc[0].to_dict()
    for key, value in list(row.items()):
        if pd.isna(value):
            row[key] = None
        elif hasattr(value, "item"):
            row[key] = value.item()
    return {"sample_id": sample_id, "record": row}
