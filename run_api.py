"""Ponto de entrada local da API de campo (porta 8001 para não colidir com o Modelo 1)."""
import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)
