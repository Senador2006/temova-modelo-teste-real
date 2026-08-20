# Guia de campo — medição real D0 → D2 → D4

Roteiro por equipe. Objetivo: **não misturar datas, não vazar D4 e não perder a previsão**.

Modelo em uso: **2.3.0**

```text
D0 -------- 2 dias -------- D2 -------- 2 dias -------- D4
 contexto                    GERAR A PREVISÃO            só validar
```

Se atrasar um dia, o experimento quebra. D0→D2 e D2→D4 precisam ser **2 dias cada**, no **mesmo horário** (±1 h).

| Fase | Exemplo |
|---|---|
| **D0** — contexto | sexta 09:00 |
| **D2** — previsão (salvar hoje) | domingo 09:00 |
| **D4** — validação | terça 09:00 |

---

## Quem faz o quê

| Equipe | Responsabilidade | Não faz |
|---|---|---|
| **Campo** | Marcar área, medir 5 pontos, fotos, horários | Clima da API, rodar o modelo |
| **Coleta e API** | Clima observado, forecast D2→D4, preencher CSVs de clima | Medir gramínea, retreinar a rede |
| **Desenvolvimento** | Montar a janela LSTM, gerar e persistir a previsão, calcular erro | Ir ao trecho no D4 “para corrigir” o input |

**Passagem de bastão**

```text
Campo  --alturas + data/hora-->  Coleta/API  --CSV completo-->  Desenvolvimento
                                      ^
                                      |
                               clima observado + forecast
```

No D2 a previsão precisa existir **antes** de qualquer um ir embora. Sem isso, o teste real não vale.

---

# Equipe de campo

Vocês medem a vegetação. Tudo em **centímetros**. Os 5 pontos não mudam depois do D0.

## Material

- [ ] Régua ou trena em cm (a **mesma** em D0, D2 e D4)
- [ ] Estacas / fita / tinta para área e 5 pontos
- [ ] Celular com relógio e câmera
- [ ] Caderno ou a folha da seção final deste guia
- [ ] Combinar calendário com as outras duas equipes **antes**

## Como marcar a área (só no D0)

1. Escolher trecho de gramínea **homogêneo**, sem poda no meio do experimento.
2. Marcar um retângulo fixo.
3. Marcar **5 pontos permanentes** (P1 a P5). Não trocar depois.
4. Tirar foto de referência.

Em cada visita, medir no **mesmo ponto**, da base do solo até o topo da folha/haste dominante daquele ponto. Não escolher “a folha mais alta do trecho” de forma diferente a cada dia.

```text
altura_media = (P1 + P2 + P3 + P4 + P5) / 5
```

Essa média é a única altura que o modelo usa. Entregar o valor para a equipe de coleta no mesmo dia.

## D0

1. [ ] Chegar no horário combinado.
2. [ ] Confirmar área e P1…P5 marcados.
3. [ ] Medir P1…P5 em cm.
4. [ ] Calcular `altura_media` (altura_D0).
5. [ ] Anotar data/hora exata.
6. [ ] Foto da área.
7. [ ] Passar médias + data/hora para Coleta/API.

Não esperem o modelo. Sem D2 a janela está incompleta.

## D2

1. [ ] Mesmos 5 pontos, mesma régua, mesmo horário.
2. [ ] Medir P1…P5 e calcular `altura_media` (altura_D2).
3. [ ] Anotar data/hora.
4. [ ] Foto de conferência.
5. [ ] Passar os números **no mesmo dia** para Coleta/API e Desenvolvimento.
6. [ ] Confirmar com Desenvolvimento que a previsão foi salva antes de encerrar o D2.

## D4

1. [ ] Mesmos 5 pontos, mesma régua, mesmo horário.
2. [ ] Medir P1…P5 e calcular `altura_media` (altura_D4).
3. [ ] Anotar data/hora.
4. [ ] Passar a média para Coleta/API e Desenvolvimento.

Não mudem a área. Não escolham pontos novos para “ficar mais bonito”.

## O que o campo nunca faz

- Medir uma folha diferente a cada dia, ou só o ponto mais alto.
- Trocar de área no meio do experimento.
- Usar milímetros. Vegetação é **cm**.
- Fazer D0 na segunda e D2 na terça (isso é 1 dia, não 2).
- Esperar o D4 para “repetir a medição do D2”.

## Folha de campo

```text
Área / sample_id: ____________________
Local: _______________________________

D0  data/hora: ____ / ____ / ______  __:__
P1 ____  P2 ____  P3 ____  P4 ____  P5 ____   média D0 = ______ cm

D2  data/hora: ____ / ____ / ______  __:__
P1 ____  P2 ____  P3 ____  P4 ____  P5 ____   média D2 = ______ cm

D4  data/hora: ____ / ____ / ______  __:__
P1 ____  P2 ____  P3 ____  P4 ____  P5 ____   média D4 = ______ cm
```

---

# Equipe de coleta e dados de API

Vocês fecham o clima e o CSV bruto. Sem o forecast certo no D2, a LSTM recebe o input errado.

Arquivo de vocês:

`data/templates/real_field_collection_template.csv`

Uma linha por fase: `D0_CONTEXTO`, `D2_PREVISAO`, `D4_VALIDACAO`.

## O que buscar

| Dado | Unidade | D0 | D2 | D4 |
|---|---|---|---|---|
| Clima **observado**: temp média | °C | sim | sim | registro apenas |
| Clima **observado**: precipitação | mm | sim | sim | registro apenas |
| Clima **observado**: umidade | % | sim | sim | registro apenas |
| `fonte_clima` (API/app) | texto | sim | sim | sim |
| Forecast D2→D4: temp **média** | °C | não | **obrigatório** | não |
| Forecast D2→D4: chuva **acumulada** | mm | não | **obrigatório** | não |
| Forecast D2→D4: umidade **média** | % | não | **obrigatório** | não |

No D2, o forecast é o que a API mostrava **naquele momento**. Não substituir depois pelo clima que de fato ocorreu.

## D0

1. [ ] Receber de Campo: 5 pontos, média, data/hora.
2. [ ] Consultar API/app: temperatura, precipitação e umidade **observadas**.
3. [ ] Anotar a fonte (`OpenWeather`, `INMET`, etc.).
4. [ ] Preencher a linha `D0_CONTEXTO`.
5. [ ] Conferir unidades: °C, mm, %.

## D2

1. [ ] Receber de Campo a média D2 e a data/hora.
2. [ ] Consultar clima **observado** de hoje.
3. [ ] Consultar **agora** a previsão dos próximos 2 dias:
   - `temp_futuro_2d_c` = média prevista
   - `precipitacao_futuro_2d_mm` = acumulado previsto
   - `umidade_futuro_2d_pct` = média prevista
4. [ ] Guardar um print/export da consulta (prova de que o forecast é de D2).
5. [ ] Preencher a linha `D2_PREVISAO`.
6. [ ] Entregar o CSV para Desenvolvimento **ainda no D2**.

## D4

1. [ ] Receber de Campo a média D4.
2. [ ] Registrar clima observado de D4 se quiser (auditoria).
3. [ ] **Não** atualizar o forecast do D2 com o que realmente choveu.
4. [ ] Preencher a linha `D4_VALIDACAO` (alturas + data/hora).
5. [ ] Entregar `altura_real_d4_cm` para Desenvolvimento calcular o erro.

## O que a coleta/API nunca faz

- Usar o clima real de D3/D4 como se fosse o forecast de D2.
- Mudar a fonte da API no meio do experimento sem anotar.
- Entregar só “choveu / não choveu”. O modelo precisa de número.
- Preencher `altura_real_d4_cm` no template da LSTM **antes** da previsão existir.

---

# Equipe de desenvolvimento

Vocês ligam Campo + API no modelo. A prova de conceito só existe se a previsão for salva no **D2**.

Arquivos:

| Arquivo | Quando |
|---|---|
| `data/templates/real_lstm_input_template.csv` | D2 — janela da rede |
| `results/predicoes_reais.csv` | D2 — previsão; D4 — erro |
| `artifacts/lstm_model.keras` + scalers | já treinados; não reajustar |
| `data/data_dictionary.csv` | dúvida de coluna |
| `README.md` | spec do modelo |

## D0

1. [ ] Confirmar que o modelo **2.3.0** e os scalers estão em `artifacts/`.
2. [ ] Não treinar de novo. Não fit de scaler.
3. [ ] Aguardar D2. Sem o segundo timestep não há input.

## D2 — gerar e persistir a previsão

Preencher **somente** o que o modelo pode ver hoje:

| Coluna | De onde sai |
|---|---|
| `sample_id` | ex. `REAL001` |
| `altura_t1_cm` … `umidade_t1_pct` | Campo + API do **D0** |
| `altura_t2_cm` … `umidade_t2_pct` | Campo + API do **D2** |
| `temp_futuro_2d_c` | forecast de D2 (Coleta/API) |
| `precipitacao_futuro_2d_mm` | forecast de D2, acumulado |
| `umidade_futuro_2d_pct` | forecast de D2 |
| `crescimento_passado_2d_cm` | `altura_D2 - altura_D0` |
| `data_previsao` | data de **hoje (D2)** |

**Deixar em branco no D2:**

- `crescimento_futuro_2d_cm`
- `altura_real_d4_cm`
- `crescimento_real_2d_cm`
- `crescimento_previsto_2d_cm`
- `erro_absoluto_cm`

```powershell
cd modelo_teste_real
$env:KERAS_BACKEND = "torch"
..\venv\Scripts\python.exe predict.py --input data/templates/real_lstm_input_template.csv --output results/predicoes_reais.csv
```

1. [ ] Abrir `results/predicoes_reais.csv`.
2. [ ] Anotar `crescimento_previsto_2d_cm` no caderno **e** no CSV.
3. [ ] Anotar o baseline: `altura_D2 - altura_D0`.
4. [ ] Não apagar nem sobrescrever esse arquivo.
5. [ ] Avisar Campo e Coleta que a previsão de D2 está persistida.

Se o modelo só rodar no D4, o experimento perde o valor de prova de conceito.

O que entra na LSTM:

```text
[altura_D0, temp_D0, chuva_D0, umidade_D0]
[altura_D2, temp_D2, chuva_D2, umidade_D2]
+ forecast D2→D4
```

O que **não** entra: `altura_D4`, clima real de D3/D4, crescimento real.

## D4 — só erro

Não regenere a previsão com o clima que aconteceu. Use o número salvo no D2.

```text
crescimento_real_2d_cm = altura_D4 - altura_D2
erro_absoluto_cm       = |previsto_em_D2 - crescimento_real|
erro_baseline_cm       = |(altura_D2 - altura_D0) - crescimento_real|
```

1. [ ] Colocar `altura_real_d4_cm` no CSV e rodar de novo o `predict.py`, **ou** calcular na mão.
2. [ ] Comparar LSTM vs baseline.
3. [ ] Com um único ponto, o número que vale é o **erro absoluto em cm**.
4. [ ] Não retreinar com essa amostra. Ela é teste externo.

## O que o desenvolvimento nunca faz

- Esperar o D4 para gerar a previsão.
- Retreinar modelo ou scalers com o dado real.
- Preencher `altura_real_d4_cm` antes de salvar o previsto.
- Trocar o forecast de D2 pelo clima observado depois.
- Mudar a ordem das features na mão.

---

# Encerramento do experimento

O teste real só está completo quando:

1. Campo mediu D0, D2 e D4 nos mesmos 5 pontos.
2. Coleta/API registrou clima observado e o **forecast de D2**.
3. Desenvolvimento salvou a previsão **no D2**.
4. D4 foi medido depois, sem alterar o input.
5. Erro da LSTM e do baseline foram calculados.

Se houver mais de uma área: `REAL001`, `REAL002`, … — nunca misturar pontos entre áreas.
