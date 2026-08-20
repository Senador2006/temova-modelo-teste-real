# Melhorias e Correções --- Modelo de Previsão de Crescimento de Gramínea

## 1. Objetivo do projeto

O projeto tem como objetivo prever o **crescimento, em centímetros, de
uma vegetação gramínea nos próximos 2 dias**.

A estratégia definida é:

1.  treinar inicialmente o modelo com dados sintéticos;
2.  utilizar uma janela real de 2 dias como contexto;
3.  obter dados climáticos observados por API;
4.  obter a previsão climática dos 2 dias seguintes por API;
5.  gerar a previsão de crescimento;
6.  aguardar 2 dias;
7.  medir novamente a vegetação;
8.  comparar crescimento previsto e crescimento real.

O teste real deve permanecer independente do treinamento sintético.

------------------------------------------------------------------------

## 2. Fluxo temporal correto

A principal correção necessária é garantir que **treinamento, validação,
teste sintético e experimento de campo representem exatamente a mesma
escala temporal**.

O fluxo oficial deve ser:

``` text
             CONTEXTO REAL / SINTÉTICO
             <------ 2 dias ------>

D0 -------------------------------- D2
 |                                   |
altura                              altura
clima                               clima
 |                                   |
 +---------------+-------------------+
                 |
                 v
               LSTM
                 ^
                 |
        previsão climática
        para D2 até D4
                 |
                 v
        crescimento previsto
                 |
                 |
D2 -------------------------------- D4
             <------ 2 dias ------>
                 |
                 v
        crescimento observado
                 |
                 v
         cálculo das métricas
```

Portanto:

-   `D0 → D2` = janela de contexto de **2 dias**;
-   `D2 → D4` = horizonte de previsão de **2 dias**.

------------------------------------------------------------------------

## 3. Correção crítica: inconsistência temporal

### Problema identificado

O dataset/modelo atual trata `t1 → t2` como aproximadamente **1 dia** em
partes da geração sintética.

Isso aparece, por exemplo, na variável:

``` text
crescimento_passado_1d_cm
```

e na lógica anterior do baseline:

``` text
baseline = 2 * (altura_t2 - altura_t1)
```

Entretanto, o experimento real foi definido como:

``` text
D0 -------- 2 dias -------- D2
```

Logo:

``` text
altura_D2 - altura_D0
```

representa crescimento em **2 dias**, não em 1 dia.

### Correção obrigatória

Todo o pipeline deve passar a representar:

``` text
D0 → D2 = 2 dias
D2 → D4 = 2 dias
```

O dataset sintético precisa ser regenerado respeitando essa estrutura.

------------------------------------------------------------------------

## 4. Correção da nomenclatura

Substituir:

``` text
crescimento_passado_1d_cm
```

por:

``` text
crescimento_passado_2d_cm
```

Definição:

``` text
crescimento_passado_2d_cm =
    altura_t2_cm - altura_t1_cm
```

ou, na nomenclatura experimental:

``` text
crescimento_passado_2d_cm =
    altura_D2 - altura_D0
```

Essa variável pode ser usada para análise, baseline e eventualmente como
feature derivada.

------------------------------------------------------------------------

## 5. Correção do gerador sintético

Cada amostra sintética deve reproduzir o mesmo experimento que será
realizado em campo.

### Estrutura esperada

``` text
D0
|
| 2 dias simulados
|
D2
|
| 2 dias simulados
|
D4
```

O gerador deve produzir:

### Contexto

-   altura em D0;
-   clima referente à janela D0→D2;
-   altura em D2.

### Futuro

-   previsão climática correspondente a D2→D4;
-   altura em D4.

### Target

``` text
crescimento_futuro_2d_cm =
    altura_D4 - altura_D2
```

A função responsável pela simulação do crescimento entre D0 e D2 deve
usar explicitamente um intervalo de **2 dias**.

Se houver algo equivalente a:

``` python
growth_d0_d2 = growth_cm(..., days=1)
```

deve ser corrigido para:

``` python
growth_d0_d2 = growth_cm(..., days=2)
```

desde que a função `growth_cm` esteja definida para receber a duração
dessa forma.

------------------------------------------------------------------------

## 6. Correção do baseline

O baseline anterior considerava:

``` text
baseline = 2 × (altura_t2 - altura_t1)
```

Isso fazia sentido somente quando `t1 → t2` representava 1 dia.

Com a nova estrutura:

``` text
D0 → D2 = 2 dias
```

e:

``` text
D2 → D4 = 2 dias
```

o baseline correto passa a ser:

``` text
baseline =
    altura_D2 - altura_D0
```

Interpretação:

> Se a vegetação cresceu X centímetros nos últimos 2 dias, o baseline
> assume que crescerá aproximadamente X centímetros nos próximos 2 dias.

A LSTM deve ser comparada contra esse baseline.

------------------------------------------------------------------------

## 7. Estrutura de entrada da LSTM

A arquitetura atual pode ser preservada inicialmente.

### Entrada sequencial

Shape:

``` text
(N, 2, 4)
```

Cada timestep contém:

``` text
[
    altura_cm,
    temperatura_media,
    precipitacao_mm,
    umidade_media
]
```

Representação:

``` text
X_sequence = [
    [altura_D0, temp_D0, chuva_D0, umidade_D0],
    [altura_D2, temp_D2, chuva_D2, umidade_D2]
]
```

### Observação

Com apenas dois timesteps, a LSTM possui uma sequência extremamente
curta.

Isso não impede seu uso no MVP, mas torna obrigatória posteriormente a
comparação com modelos tabulares.

------------------------------------------------------------------------

## 8. Entrada climática futura

A previsão meteorológica dos próximos 2 dias deve continuar sendo
fornecida separadamente.

Shape:

``` text
(N, 3)
```

Features:

``` text
temp_futuro_2d_c
precipitacao_futuro_2d_mm
umidade_futuro_2d_pct
```

Preferencialmente:

-   temperatura = média prevista para D2→D4;
-   precipitação = acumulado previsto para D2→D4;
-   umidade = média prevista para D2→D4.

A previsão utilizada deve ser aquela **disponível em D2**, no momento em
que o modelo gera sua previsão.

------------------------------------------------------------------------

## 9. Regra contra data leakage

Nunca utilizar o clima que efetivamente ocorreu em D2→D4 para produzir a
previsão feita em D2.

Correto:

``` text
D2
|
+-- consulta API
|
+-- previsão meteorológica D2→D4
|
+-- modelo gera previsão
|
D4
|
+-- verifica o que realmente aconteceu
```

Incorreto:

``` text
D4
|
+-- consulta clima observado D2→D4
|
+-- usa esse clima para reconstruir a previsão de D2
```

O segundo caso introduziria informação futura no modelo.

------------------------------------------------------------------------

## 10. Target oficial

O target do projeto permanece:

``` text
crescimento_futuro_2d_cm
```

Definido como:

``` text
crescimento_futuro_2d_cm =
    altura_D4 - altura_D2
```

Exemplo:

``` text
altura_D2 = 8,20 cm
altura_D4 = 8,71 cm
```

Logo:

``` text
crescimento_real = 0,51 cm
```

O modelo deve tentar prever `0,51 cm`.

------------------------------------------------------------------------

## 11. Medição real da vegetação

Evitar utilizar uma única folha ou ponto como representação da área.

Em cada momento --- D0, D2 e D4 --- medir preferencialmente **5 pontos
fixos ou padronizados**.

Exemplo:

``` text
P1 = 7,8 cm
P2 = 8,1 cm
P3 = 7,9 cm
P4 = 8,0 cm
P5 = 8,2 cm
```

Utilizar:

``` text
altura_media =
    (P1 + P2 + P3 + P4 + P5) / 5
```

Essa média passa a ser a `altura_cm` usada pelo modelo.

O procedimento deve permanecer o mais consistente possível entre D0, D2
e D4.

------------------------------------------------------------------------

## 12. Dados climáticos mínimos

Para o MVP:

### Obrigatórios

-   temperatura média;
-   precipitação;
-   umidade relativa.

### Vegetação

-   altura média.

### Futuro

-   temperatura média prevista para os próximos 2 dias;
-   precipitação acumulada prevista para os próximos 2 dias;
-   umidade média prevista para os próximos 2 dias.

Variáveis como radiação solar podem ser adicionadas posteriormente caso
a API escolhida forneça dados confiáveis e consistentes.

------------------------------------------------------------------------

## 13. Arquitetura inicial

A arquitetura atual pode ser mantida:

``` text
Input sequência (2 × 4)
        |
        v
     LSTM(32)
        |
        +----------------+
                         |
Input clima futuro (3)  |
        |                |
        +---- Concatenate
                 |
                 v
              Dense(16)
                 |
                 v
               Dense(1)
                 |
                 v
       crescimento em cm
```

Não há necessidade de aumentar a complexidade da rede antes do teste
real.

------------------------------------------------------------------------

## 14. Normalização

Manter `StandardScaler`.

Regra:

``` text
synthetic_train
       |
       +-- fit scaler
       |
       v
scaler treinado
       |
       +-- transform validation
       +-- transform synthetic test
       +-- transform real field data
```

Nunca executar `fit` novamente utilizando o experimento real.

Os scalers devem ser salvos junto com o modelo.

------------------------------------------------------------------------

## 15. Divisão dos dados sintéticos

Manter uma separação semelhante a:

``` text
70% treinamento
15% validação
15% teste sintético
```

O conjunto real não faz parte dessa divisão.

Ele constitui um **teste externo independente**.

------------------------------------------------------------------------

## 16. Métricas

### Principal

**MAE**

Interpretação direta:

> Em média, o modelo erra X centímetros.

### Complementares

-   RMSE;
-   R².

Para um único experimento real, utilizar principalmente:

``` text
erro_absoluto =
|crescimento_previsto - crescimento_real|
```

Exemplo:

``` text
Previsto: 0,58 cm
Real:     0,51 cm

Erro absoluto = 0,07 cm
```

------------------------------------------------------------------------

## 17. Comparação obrigatória com baseline

O resultado da LSTM isoladamente não é suficiente.

Sempre comparar:

``` text
Baseline
   VS
LSTM
```

Posteriormente:

``` text
Baseline
   VS
Random Forest / XGBoost
   VS
LSTM
```

Com apenas dois timesteps, um modelo tabular pode apresentar desempenho
igual ou superior à LSTM.

Isso não representa falha do projeto; é um resultado experimental
relevante.

------------------------------------------------------------------------

## 18. Limitação dos dados sintéticos

Os dados sintéticos são úteis para:

-   desenvolver o pipeline;
-   validar código;
-   testar arquitetura;
-   testar normalização;
-   testar persistência do modelo;
-   verificar o processo de inferência.

Eles **não provam** que o modelo aprendeu o comportamento biológico real
da vegetação.

Mesmo um resultado excelente no teste sintético significa apenas:

> O modelo aprendeu corretamente as relações presentes no simulador.

A verdadeira avaliação começa quando o modelo recebe D0/D2 reais e tenta
prever D4.

------------------------------------------------------------------------

## 19. Domain shift

O principal risco do projeto é a diferença entre:

``` text
mundo sintético
```

e:

``` text
vegetação real
```

Fatores não representados podem afetar o crescimento:

-   espécie;
-   solo;
-   disponibilidade real de água;
-   poda;
-   pisoteio;
-   pragas;
-   luminosidade local;
-   nutrientes;
-   erro de medição.

Esses fatores não precisam ser incluídos no MVP, mas devem constar como
limitações.

------------------------------------------------------------------------

## 20. Experimento oficial

### D0

-   marcar a área;
-   realizar as medições;
-   calcular altura média;
-   registrar data/hora;
-   obter clima via API.

### D2

-   repetir as medições;
-   calcular altura média;
-   obter clima da janela de contexto;
-   consultar previsão meteorológica D2→D4;
-   montar entrada;
-   gerar previsão;
-   salvar imediatamente a previsão.

### D4

-   repetir as medições;
-   calcular altura média;
-   calcular crescimento real:

``` text
altura_D4 - altura_D2
```

-   comparar com a previsão;
-   calcular erro absoluto.

------------------------------------------------------------------------

## 21. Prioridade das correções

### P0 --- antes de qualquer novo treinamento

-   [x] Corrigir D0→D2 para representar 2 dias no simulador.
-   [x] Corrigir D2→D4 para representar 2 dias.
-   [x] Regenerar dataset sintético.
-   [x] Renomear `crescimento_passado_1d_cm` para
    `crescimento_passado_2d_cm`.
-   [x] Corrigir baseline.
-   [x] Retreinar scalers.
-   [x] Retreinar LSTM.
-   [x] Gerar novamente métricas de validação e teste.

### P1 --- antes do teste de campo

-   [x] Confirmar unidade de todas as features.
-   [x] Confirmar ordem das features usada no treinamento e inferência.
-   [ ] Definir API meteorológica.
-   [ ] Implementar consulta de histórico climático.
-   [ ] Implementar consulta da previsão D2→D4.
-   [x] Testar pipeline de inferência completo.
-   [x] Garantir que previsão e timestamp sejam persistidos em D2.

### P2 --- após o primeiro teste real

-   [ ] Comparar LSTM e baseline.
-   [ ] Testar Random Forest ou XGBoost.
-   [ ] Repetir experimento real em novas janelas.
-   [ ] Avaliar domain shift.
-   [ ] Considerar calibração com dados reais somente depois de
    preservar um conjunto real independente para teste.

------------------------------------------------------------------------

## 22. Critério de pronto para campo

O modelo será considerado pronto para o primeiro experimento quando:

1.  treino e campo utilizarem exatamente a janela D0→D2;
2.  o target representar exatamente D2→D4;
3.  o dataset sintético tiver sido regenerado;
4.  baseline estiver corrigido;
5.  scalers forem ajustados apenas no treino;
6.  o modelo e os scalers estiverem salvos;
7.  o pipeline aceitar uma amostra real sem alterações manuais de
    estrutura;
8.  a previsão meteorológica utilizada for a disponível em D2;
9.  a previsão gerada em D2 for persistida antes de D4;
10. o processo de cálculo do erro estiver definido.

------------------------------------------------------------------------

## 23. Resultado esperado do MVP

O objetivo do primeiro experimento não é provar que o modelo é
universalmente preciso.

O objetivo é responder:

> **Um modelo treinado inicialmente com dados sintéticos, recebendo 2
> dias de contexto real e previsão meteorológica, consegue produzir uma
> estimativa razoável do crescimento de uma gramínea nos 2 dias
> seguintes?**

O primeiro teste de campo fornece uma prova de conceito.

Repetições posteriores permitem medir a capacidade real de generalização
do modelo.
