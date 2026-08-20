FROM python:3.13-slim

WORKDIR /app

ENV KERAS_BACKEND=torch
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/artifacts \
 && if [ ! -f artifacts/lstm_model.keras ]; then \
      echo "Artefatos ausentes — treinando (lento, só use se não versionou os pesos)."; \
      python train_model.py; \
    fi

EXPOSE 8001

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8001} --workers 1"]
