# Modelo LSTM — Previsão de crescimento de gramínea (2 dias)

MVP de **série temporal** para prever o crescimento, em centímetros, de uma gramínea nos **2 dias seguintes**. O treino usa dados sintéticos; o teste de campo (real ou simulado) **não entra no ajuste do modelo**.

Versão atual: **2.3.0**

Guia para a coleta real (não se perder no D0/D2/D4): [`GUIA_CAMPO.md`](GUIA_CAMPO.md)

A escala temporal é a mesma no treino e no campo:

```text
D0 -------- 2 dias -------- D2 -------- 2 dias -------- D4
         contexto                  horizonte
```

---

## 1. Objetivo

Em **D2**, o modelo prevê quanto a gramínea vai crescer até **D4**, usando:

- dois timesteps de contexto (D0 e D2): altura média, temperatura, precipitação e umidade;
- a **previsão climática** agregada dos próximos 2 dias, disponível no momento da previsão.

Alvo:

```text
crescimento_futuro_2d_cm = altura_D4 - altura_D2
```

A altura de D4 **nunca** entra como feature. O valor `crescimento_real_2d_cm` existe só para calcular o erro depois da medição.

---

## 2. Especificações do modelo

| Aspecto | Detalhe |
|---|---|
| Tipo | LSTM com ramo auxiliar de clima futuro |
| Versão | 2.3.0 |
| Backend | Keras 3 + PyTorch |
| Entrada sequencial | `X_sequence = (N, 2, 4)` |
| Entrada futura | `X_future = (N, 3)` |
| Saída | 1 valor: crescimento em cm (D2→D4) |
| Scalers | `StandardScaler` (sequência, clima futuro e alvo) |
| Fit dos scalers | **somente** `synthetic_lstm_train.csv` |
| Seed | `20260813` (sintético) / `20260814` (campo simulado) |
| Otimizador / loss | Adam / MSE |
| Métrica de treino | MAE |
| Early stopping | `val_mae`, patience 12, restore best weights |

### 2.1 Features

**Sequência (por timestep), nesta ordem:**

1. `altura_cm`
2. `temp_media_c`
3. `precipitacao_mm`
4. `umidade_media_pct`

```text
[
  [altura_t1, temp_t1, chuva_t1, umidade_t1],   # D0
  [altura_t2, temp_t2, chuva_t2, umidade_t2]    # D2
]
```

**Clima futuro (agregado dos próximos 2 dias, previsto em D2):**

1. `temp_futuro_2d_c`
2. `precipitacao_futuro_2d_mm`
3. `umidade_futuro_2d_pct`

### 2.2 Arquitetura

```text
Input sequência (2 × 4)
        |
      LSTM(32)
        |
        +--------------+
                       |
Input clima futuro (3) |
        |              |
        +--- Concatenate
                |
             Dense(16, ReLU)
                |
              Dense(1)
                |
      crescimento em cm
```

### 2.3 Baseline obrigatório

Antes de confiar na LSTM, comparar com:

```text
crescimento_previsto_baseline = altura_D2 - altura_D0
                              = altura_t2_cm - altura_t1_cm
```

Interpretação: se a vegetação cresceu X cm nos últimos 2 dias, o baseline assume X cm nos próximos 2 dias. Não se multiplica por 2 — as duas janelas já têm a mesma duração.

### 2.4 Métricas

Prioridade:

1. **MAE** — erro médio em centímetros (métrica principal).
2. **RMSE** — penaliza erros grandes.
3. **R²** — complementar; com poucos testes de campo é pouco informativa.

Para um único ponto real, o número que importa é o **erro absoluto em cm**.

### 2.5 Regras anti-leakage

- D4 não entra como feature da previsão feita em D2.
- O clima futuro usado em D2 é a **previsão disponível naquele momento**, não o clima observado depois.
- Dados de campo (reais ou simulados como holdout) **não** ajustam modelo nem scalers.
- `StandardScaler` é fit só no treino sintético e reaplicado com `transform`.

---

## 3. Como funciona

```text
dados sintéticos          campo (simulado ou real)
        |                           |
   train_model.py              D0 medir 5 pontos
        |                      D2 medir + forecast
        v                           |
  artifacts/ (modelo + scalers)     v
        |                    predict.py
        +-------------------------->|
                                    |
                              previsão D2→D4
                                    |
                              D4 medir (holdout)
                                    |
                         erro = |previsto - real|
                         vs baseline
```

### Experimento D0 → D2 → D4

**D0 — início do contexto**

1. Marcar uma área fixa.
2. Medir 5 pontos e calcular a altura média.
3. Registrar data/hora e clima observado.

**D2 — momento da previsão**

1. Repetir as 5 medições.
2. Obter clima observado e a previsão dos próximos 2 dias.
3. Preencher o CSV de entrada (sem D4).
4. Rodar `predict.py` com os scalers do treino.
5. Registrar `crescimento_previsto_2d_cm`.

**D4 — validação**

1. Medir de novo.
2. `crescimento_real_2d_cm = altura_D4 - altura_D2`
3. `erro_absoluto_cm = |previsto - real|`

---

## 4. Organização de arquivos

```text
modelo_teste_real/
├── README.md                          # este documento
├── requirements.txt
├── generate_synthetic_data.py         # regenera o dataset D0→D2 / D2→D4
├── train_model.py                     # treino sintético + métricas vs baseline
├── predict.py                         # inferência D2→D4 (+ erro se D4 existir)
├── simulate_field_data.py             # gera o ensaio de campo simulado
├── ml/
│   ├── config.py                      # caminhos, features e hiperparâmetros
│   ├── growth_simulator.py            # crescimento de 2 dias (sintético e campo)
│   ├── preprocessing.py               # tensors, scalers, persistência
│   ├── model_builder.py               # LSTM(32) + clima futuro
│   └── metrics.py                     # MAE, RMSE, R²
├── data/
│   ├── data_dictionary.csv            # dicionário de colunas
│   ├── synthetic/                     # 10.000 amostras (70/15/15)
│   │   ├── synthetic_lstm_full.csv
│   │   ├── synthetic_lstm_train.csv   # 7.000 — único split com fit
│   │   ├── synthetic_lstm_validation.csv
│   │   └── synthetic_lstm_test.csv
│   ├── templates/                     # formulários para coleta real
│   │   ├── real_field_collection_template.csv
│   │   └── real_lstm_input_template.csv
│   └── simulated_field/               # holdout que imita o campo
│       ├── simulated_field_collection.csv
│       └── simulated_real_lstm_input.csv
├── artifacts/                         # modelo e scalers (não reajustar)
│   ├── lstm_model.keras
│   ├── scaler_sequence.joblib
│   ├── scaler_future.joblib
│   ├── scaler_target.joblib
│   ├── metadata.json
│   └── metrics.json                   # resultado do teste sintético
└── results/                           # saídas de previsão
    └── predicoes_campo_simulado.csv
```

Scripts ficam na raiz do módulo. Dados, pesos e resultados não se misturam.

---

## 5. Testes e resultados

### 5.1 Teste sintético (in-distribution)

Holdout de 1.500 amostras em `data/synthetic/synthetic_lstm_test.csv`. Não usado no fit dos scalers nem na escolha implícita via early stopping (essa usa a validação).

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| **LSTM** | **0.036 cm** | **0.044 cm** | **0.88** |
| Baseline | 0.124 cm | 0.162 cm | −0.68 |

- Treino: 7.000 amostras, validação 1.500.
- Early stopping na epoch 20 (máximo 80).
- Altura em D0: média ~62,1 cm (faixa 40–90 cm), alinhada à coleta real REAL001 (média 62,25 cm; pontos 53–80 cm).
- `crescimento_passado_2d_cm` e `crescimento_futuro_2d_cm` têm média ~0,26 cm — gramado maduro/alto cresce bem menos que a faixa recém-podada (~1 cm / 2 dias) da v2.2.0.
- A LSTM superou o baseline persistente (MAE cerca de 1/3).

Bom resultado aqui **não prova** desempenho em campo: o sintético só valida o pipeline.

### 5.2 Teste de campo simulado (holdout)

10 áreas (`REAL001`–`REAL010`) geradas por `simulate_field_data.py`, com seed `20260814` (diferente do treino).

O ensaio imita o protocolo real:

- 5 pontos **fixos** por área (D0, D2, D4), com heterogeneidade espacial e ruído de medição;
- alturas na faixa do gramado alto da coleta REAL001 (~45–80 cm);
- em D2 a LSTM recebe o **forecast**, não o clima que de fato ocorreu depois;
- `crescimento_real_2d_cm` e `altura_real_d4_cm` são holdout.

| Área | Previsto LSTM | Real D4 | Erro absoluto |
|---|---|---|---|
| REAL001 | 0.217 cm | 0.344 cm | 0.127 |
| REAL002 | 0.571 cm | 0.604 cm | 0.033 |
| REAL003 | 0.167 cm | 0.266 cm | 0.099 |
| REAL004 | 0.354 cm | 0.295 cm | 0.059 |
| REAL005 | 0.277 cm | 0.151 cm | 0.126 |
| REAL006 | 0.132 cm | 0.326 cm | 0.194 |
| REAL007 | 0.258 cm | 0.259 cm | 0.001 |
| REAL008 | 0.405 cm | 0.319 cm | 0.086 |
| REAL009 | 0.173 cm | 0.235 cm | 0.063 |
| REAL010 | 0.504 cm | 0.458 cm | 0.046 |

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| **LSTM** | **0.083 cm** | **0.099 cm** | 0.32 |
| Baseline | 0.180 cm | 0.206 cm | −1.95 |

A LSTM continua melhor que o baseline. Com 10 pontos, ruído de medição e erro de forecast, o número que importa aqui é o MAE em cm.

Detalhe das previsões: `results/predicoes_campo_simulado.csv`.

### 5.3 O que ainda falta para o MVP de campo real

O critério mínimo de sucesso só fecha quando:

- a previsão é feita em D2 **antes** de conhecer D4;
- o valor previsto fica registrado;
- D4 é medido fisicamente;
- o erro absoluto é calculado;
- o resultado é comparado com o baseline.

---

## 6. Como executar

Pré-requisito: Python 3.13+ (ou 3.14) com o `venv` da raiz do repositório. TensorFlow não é necessário; o backend é PyTorch.

```powershell
cd modelo_teste_real
..\venv\Scripts\pip.exe install -r requirements.txt
$env:KERAS_BACKEND = "torch"
```

### Regenerar o dataset sintético (janela 2+2 dias)

```powershell
..\venv\Scripts\python.exe generate_synthetic_data.py
```

### Treinar

```powershell
..\venv\Scripts\python.exe train_model.py
..\venv\Scripts\python.exe train_model.py --epochs 80 --batch-size 64
```

Gera `artifacts/lstm_model.keras` e os três scalers.

### Gerar o ensaio de campo simulado

```powershell
..\venv\Scripts\python.exe simulate_field_data.py
```

### Prever (e comparar com D4 se a coluna existir)

```powershell
..\venv\Scripts\python.exe predict.py
..\venv\Scripts\python.exe predict.py --input data/simulated_field/simulated_real_lstm_input.csv --output results/predicoes_campo_simulado.csv
```

Quando existirem medições reais, seguir o [`GUIA_CAMPO.md`](GUIA_CAMPO.md). Resumo: preencher `data/templates/real_lstm_input_template.csv` no **D2** (sem D4) e rodar:

```powershell
..\venv\Scripts\python.exe predict.py --input data/templates/real_lstm_input_template.csv --output results/predicoes_reais.csv
```

A previsão precisa ser salva no D2. Depois de medir D4, incluir `altura_real_d4_cm`. O script calcula `crescimento_real_2d_cm` e `erro_absoluto_cm`.

---

## 7. Limitações

O dataset sintético e o ensaio de campo simulado são **coerentes para desenvolver o pipeline**, não uma verdade biológica. As relações entre temperatura, água, umidade, altura e crescimento são simplificadas.

Desempenho bom no sintético ou no campo simulado não substitui o experimento D0–D4 real. Esse experimento é o que mede o `domain shift`.

Dicionário de colunas: [`data/data_dictionary.csv`](data/data_dictionary.csv).
