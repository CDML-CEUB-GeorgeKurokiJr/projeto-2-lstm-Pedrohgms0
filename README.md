<div align="center">

# Previsão de Ações do Setor de Energia com LSTM

**Modelo LSTM multivariado para previsão de retorno diário em uma carteira NYSE de energia**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Concluído-1D9E75?style=flat-square)
![Treino](https://img.shields.io/badge/Treino-2010–2019-534AB7?style=flat-square)
![Teste](https://img.shields.io/badge/Teste-2023–2025-534AB7?style=flat-square)

<br>

Treinado com dados de 2010–2019 e avaliado out-of-sample em 2023–2025 contra buy & hold,  
com análise de robustez em **50 seeds independentes**.

</div>

---

## Hipótese de trabalho

A modelagem conjunta de ativos correlacionados verticalmente na cadeia de exploração de petróleo — combinada com features técnicas, macroeconômicas e função de perda híbrida — permite capturar relações não-lineares que uma estratégia passiva (buy & hold) não explora. A hipótese operacional é que o modelo gera alpha estatisticamente positivo em período out-of-sample estritamente separado do treino.

---

## Resultados

> Carteira equal-weight · XOM + CVX + SLB + HAL · Teste 2023–2025

<table>
<tr>
<th align="left">Métrica</th>
<th align="center">LSTM</th>
<th align="center">Buy & Hold</th>
<th align="center">Diferença</th>
</tr>
<tr>
<td><b>Retorno total</b></td>
<td align="center">+32,63%</td>
<td align="center">−9,52%</td>
<td align="center"><b>+42,15pp</b></td>
</tr>
<tr>
<td><b>Sharpe ratio</b></td>
<td align="center"><b>0,557</b></td>
<td align="center">−0,135</td>
<td align="center">+0,692</td>
</tr>
<tr>
<td><b>Max drawdown</b></td>
<td align="center">−14,92%</td>
<td align="center">−33,27%</td>
<td align="center"><b>−18,35pp</b></td>
</tr>
<tr>
<td><b>Dias operando</b></td>
<td align="center">97,7%</td>
<td align="center">100%</td>
<td align="center">—</td>
</tr>
</table>

### Por ativo

| Ticker | LSTM | Sharpe LSTM | B&H | Sharpe B&H | MDD LSTM | MDD B&H | Alpha | Dir. Acc. |
|:------:|:----:|:-----------:|:---:|:----------:|:--------:|:-------:|:-----:|:---------:|
| **XOM** | +58,69% | 0,821 | +17,59% | 0,248 | −19,92% | −18,92% | +41,10pp | 51,7% |
| **CVX** | +31,25% | 0,491 | +1,20% | 0,019 | −17,55% | −20,64% | +30,06pp | 50,3% |
| **SLB** | +49,68% | 0,540 | −25,32% | −0,309 | −32,33% | −46,58% | +75,00pp | 49,1% |
| **HAL** | −0,74% | −0,009 | −24,59% | −0,269 | −32,11% | −53,56% | +23,84pp | 46,9% |

O modelo opera com threshold dinâmico no percentil `p30` de `|ŷ|` — fica em cash 22–38% dos dias de negociação nos sinais de baixa convicção, reduzindo exposição desnecessária ao mercado.

> **⚠️ Caveat estatístico importante:** apesar dos retornos financeiros positivos, os testes de significância não confirmam que o alpha é diferente de zero com confiança estatística. O bootstrap IC 95% do Sharpe cruza zero em todos os ativos, e o t-test do alpha não é significativo (p > 0,05 em todos os casos). A acurácia direcional filtrada também não é estatisticamente diferente de 50% pelo teste binomial. **O alpha observado pode ser ruído** e não deve ser interpretado como evidência de capacidade preditiva robusta.

> **O que os resultados não permitem afirmar:** que o modelo é lucrativo após custos reais de transação (não modelados); que o alpha é robusto a regimes adversos, dado que 2020–2022 foi excluído; que o Sharpe de 0,557 é economicamente satisfatório sem significância estatística; e que o resultado se generaliza para outros setores ou ativos.

---

## Significância Estatística

Os testes foram realizados sobre o período de teste out-of-sample (2023–2025).

### Bootstrap IC 95% do Sharpe (1.000 amostras) + t-test do alpha

| Ativo | Sharpe | IC 95% inf. | IC 95% sup. | t-test p | Sinal |
|:-----:|:------:|:-----------:|:-----------:|:--------:|:-----:|
| **XOM** | +0,821 | −0,310 | +1,977 | 0,163 | ✗ sem sinal |
| **CVX** | +0,491 | −0,687 | +1,576 | 0,405 | ✗ sem sinal |
| **SLB** | +0,540 | −0,628 | +1,702 | 0,360 | ✗ sem sinal |
| **HAL** | −0,009 | −1,193 | +1,154 | 0,988 | ✗ sem sinal |

### Métricas Estatísticas — Qualidade da Previsão

| Ativo | RMSE | MAE | R² | Dir. Acc. | p-valor (binomial) |
|:-----:|:----:|:---:|:--:|:---------:|:-----------------:|
| **XOM** | 0,0210 | 0,0163 | −1,18 | 51,7% | 0,451 |
| **CVX** | 0,0196 | 0,0151 | −0,97 | 50,3% | 0,929 |
| **SLB** | 0,0260 | 0,0193 | −0,60 | 49,1% | 0,746 |
| **HAL** | 0,0272 | 0,0201 | −0,41 | 46,9% | 0,206 |

**R² negativo** significa que o modelo performa pior do que prever a média em todos os ativos — ao contrário do esperado, onde 0,01–0,05 seria estado-da-arte (Gu et al., 2020). A **acurácia direcional filtrada** não é estatisticamente diferente de 50% em nenhum ativo (teste binomial bilateral, α = 0,05).

> **Interpretação consolidada:** os retornos financeiros positivos coexistem com ausência de significância estatística. O alpha observado pode decorrer de características específicas do período de teste (recuperação pós-pandemia de SLB/HAL, regime de mercado favorável à estratégia long/short) em vez de capacidade preditiva genuína do modelo.

---

O pipeline completo foi retreinado **50 vezes** com inicializações diferentes para verificar que os resultados não dependem de uma semente sortuda. Métricas coletadas por seed: Sharpe da carteira, capital final, alpha vs. B&H, máximo drawdown.

<div align="center">

### 🏆 LSTM bateu o buy & hold em 88% dos runs (44/50)

</div>

| Métrica | Média | Desvio | Mínimo | Máximo |
|:--------|------:|-------:|-------:|-------:|
| Sharpe (carteira) | 0,21 | 0,38 | −0,79 | **1,12** |
| Alpha vs B&H ($) | +$2.628 | $2.325 | −$2.310 | **+$8.658** |
| Capital final LSTM | $11.850 | $2.325 | $6.912 | **$17.880** |
| Max drawdown | −0,26 | 0,07 | −0,42 | −0,14 |

> Capital inicial de **$10.000** investidos.  
> Alpha médio de **+$2.628** por $10.000 investidos, com desvio padrão de $2.325.

A proporção de 88% de runs com alpha > 0 é estimativa empírica — não um teste de hipótese formal. O intervalo de confiança exato depende de suposições distribicionais que não foram verificadas.

---

## Carteira

A carteira representa uma **cadeia de valor vertical do setor de petróleo**:

| Ticker | Empresa | Subsetor | Papel |
|:------:|:-------:|:--------:|:-----:|
| **XOM** | ExxonMobil | Petróleo integrado | Âncora — benchmark global |
| **CVX** | Chevron | Petróleo integrado | Âncora — benchmark americano |
| **SLB** | Schlumberger | Serviços de exploração | Defasagem de 1–2 trimestres |
| **HAL** | Halliburton | Serviços de completação | Defasagem de 1–2 trimestres |

**Por que essa combinação?** Quando o WTI sobe, XOM e CVX lucram e aumentam o capex de exploração. Esse investimento chega a SLB e HAL com uma defasagem de 1 a 2 trimestres. A LSTM aprende essa cadeia de causalidade ao treinar os 4 ativos em conjunto — capturando tanto os movimentos comuns quanto as divergências temporais entre as integradas e os prestadores de serviço.

**Nota sobre benchmark:** o período de teste (2023–2025) coincide com recuperação pós-pandemia de SLB e HAL. Uma comparação mais exigente incluiria o ETF setorial XLE ou uma estratégia de momentum — extensão não realizada nesta versão.

---

## Dados e Split Temporal

```
┌──────────────────┬────────────────────┬──────────────────┐
│     TREINO       │     EXCLUÍDO       │      TESTE       │
│   2010 – 2019    │   2020 – 2022      │   2023 – 2025    │
│   ~2.500 dias    │   pandemia         │   ~600 dias      │
└──────────────────┴────────────────────┴──────────────────┘
```

O período **2020–2022 foi excluído intencionalmente**. A pandemia criou um regime de mercado sem precedente — o WTI ficou negativo em abril de 2020, as correlações históricas do setor quebraram e houve intervenções governamentais massivas. Treinar com esse período ensinaria ao modelo padrões que provavelmente não se repetirão.

> **Regra aplicada:** dados nunca são embaralhados. O split é estritamente temporal. O `StandardScaler` é ajustado exclusivamente no treino e aplicado ao teste sem reajuste — qualquer outra abordagem constitui data leakage.

**Limitação reconhecida:** o protocolo atual não possui conjunto de validação independente. O early stopping monitora a loss de **treino**, não de validação — o que reduz sua eficácia como regularizador. Uma partição temporal de validação (ex.: 2017–2019) fortaleceria a análise.

---

## Features — 13 Entradas em 5 Grupos Não Redundantes

A seleção foi guiada por três critérios aplicados **somente sobre o conjunto de treino**: estacionariedade verificada pelo teste ADF (p < 0,05), ausência de correlação acima de `|r| > 0,90` entre pares, e Mutual Information positiva com o target. Todas as features de preço usam log-return em vez de preço bruto.

| Feature | Grupo | Justificativa |
|:--------|:-----:|:--------------|
| `logret_close`, `logret_open` | Preço | Estacionárias (ADF p < 0,05). `logret_open` captura o gap overnight — surpresas ocorridas fora do pregão, incluindo movimentos noturnos no WTI |
| `volume_ratio` | Volume | Volume normalizado pela média de 20 dias. Detecta anomalias de participação independente do nível absoluto |
| `sma_diff`, `ema_diff` | Tendência | `sma_diff` = (SMA50 − SMA200) / SMA200 — codifica regime bull/bear institucional. `ema_diff` captura tendência de curto prazo com mais reatividade |
| `rsi14`, `macd_hist` | Momentum | RSI mede sobrecompra/sobrevenda em escala 0–100. Histograma MACD mede aceleração da tendência. Os dois têm baixa correlação entre si e são complementares |
| `atr14`, `bb_pct_b` | Volatilidade | ATR normalizado captura o regime de volatilidade, essencial num setor que alterna entre crises e booms. `bb_pct_b` combina tendência e volatilidade numa métrica 0–1 |
| `wti_logret`, `brent_logret`, `spread_wb`, `ng_logret` | Macro | WTI é o driver causal direto. Brent captura choques geopolíticos que precedem o WTI. O spread WTI–Brent reflete gargalos regionais. Gás natural tem ciclo próprio não sincronizado com o petróleo |

> **Nota metodológica:** a matriz de correlação e a Mutual Information foram calculadas somente no conjunto de treino (2010–2019). Aplicar esses critérios sobre o conjunto completo constituiria data leakage indireto.

<details>
<summary><b>Features eliminadas por redundância (|r| > 0,90)</b></summary>

<br>

| Feature eliminada | Motivo |
|:-----------------|:-------|
| `williams_r` | Correlação > 0,95 com `rsi14` — informação idêntica |
| `stoch_k`, `stoch_d` | Coberto conjuntamente por RSI + MACD |
| `CCI`, `ROC` | Derivados de variáveis já presentes |
| SMA20, SMA50, SMA200 brutos | Substituídos pelo `sma_diff` normalizado |
| `bb_width` | Redundante com `atr14` normalizado |
| `vol_hist` | Redundante com `atr14` |
| `obv_diff` | Coberto pelo `volume_ratio` |

A justificativa completa com testes ADF, matrizes de correlação e scores de Mutual Information está em `exploratory_analysis.ipynb`.

</details>

---

## Arquitetura

```
Entrada           LSTM × 2 camadas           Saída
(batch, 20, 13) ───────────────────────► Linear(128 → 1)
                  hidden = 128              sem ativação
                  dropout = 0.2
```

**Por que sem ativação na saída?** Log-returns são valores reais contínuos em ℝ — podem ser +0,03 ou −0,05. Funções como `sigmoid` ou `tanh` restringiriam a saída a intervalos fixos, introduzindo viés sistemático em retornos de maior magnitude. A camada `Linear` pura é a escolha correta para regressão de valores ilimitados.

**Por que 2 camadas LSTM?** A hierarquia permite capturar dependências temporais em múltiplas escalas — padrões de curto prazo na camada 1 e relações de prazo mais longo na camada 2.

### Função de Perda Híbrida

```
Loss = RMSE + λ · (1 − Sharpe normalizado)       λ = 0,3
```

RMSE puro minimiza o erro de previsão mas pode gerar péssimos sinais de trading — um modelo pode ter baixo RMSE e errar sistematicamente a *direção* dos movimentos, que é o que determina o resultado financeiro. O termo Sharpe penaliza estratégias com baixo retorno ajustado ao risco.

### Hiperparâmetros

| Parâmetro | Valor | Justificativa |
|:----------|:-----:|:-------------|
| Lookback (janela) | 20 dias | 1 mês de pregão — cobre RSI/SMA20 aquecidos, unidade natural de rebalanceamento institucional |
| Hidden size | 128 | Capacidade suficiente para 13 features × 4 ativos |
| Num layers | 2 | Captura padrões temporais em múltiplas escalas |
| Dropout | 0,2 | Regularização nas conexões entre camadas LSTM |
| Otimizador | Adam, lr = 1e-3 | Padrão para séries financeiras |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0,5) | Reduz LR quando a loss estagna |
| Early stopping | patience = 10, máx 100 épocas | Monitora loss de treino (ver limitação acima) |
| Gradient clipping | max_norm = 1,0 | Previne gradientes explodindo — problema clássico em LSTMs com séries voláteis |
| Batch size | 64 | |
| λ (peso do Sharpe) | 0,3 | |

---

## Threshold Dinâmico

Em vez de um valor fixo e arbitrário, cada ativo recebe seu próprio threshold calculado como o **percentil p30 da distribuição de `|ŷ|` no conjunto de treino**. Isso garante que o filtro seja proporcional à escala das previsões de cada ativo.

```
|ŷ| > threshold  e  ŷ > 0   →   long   (compra)
|ŷ| > threshold  e  ŷ < 0   →   short  (vende a descoberto)
|ŷ| ≤ threshold              →   cash   (protege capital)
```

| Ativo | Threshold calibrado | Dias operando | Dias em cash |
|:-----:|:-------------------:|:-------------:|:------------:|
| XOM | 0,004235 | 77,9% | 22,1% |
| CVX | 0,005586 | 69,3% | 30,7% |
| SLB | 0,006439 | 63,6% | 36,4% |
| HAL | 0,006601 | 62,0% | 38,0% |

O percentil p30 foi selecionado por maximização do Sharpe na distribuição de treino e aplicado **sem reajuste** no teste. Percentis abaixo de p30 colapsam para thresholds numericamente insignificantes porque a distribuição de `|ŷ|` é muito concentrada em torno de zero.

> **Risco de data snooping:** a seleção do percentil ótimo no treino introduz um grau de otimização nos hiperparâmetros da estratégia. Um procedimento mais conservador realizaria essa seleção em um conjunto de validação independente.

---

## Métricas de Avaliação

### Estatísticas — qualidade da previsão

| Métrica | Fórmula | Interpretação |
|:--------|:-------:|:--------------|
| RMSE | `√mean((ŷ − y)²)` | Penaliza erros grandes mais que pequenos |
| MAE | `mean(\|ŷ − y\|)` | Complementa o RMSE — menos sensível a outliers |
| R² | `1 − SS_res / SS_tot` | **0,01–0,05 é estado-da-arte** em previsão de retornos (Gu et al., 2020) |
| Acurácia direcional | `mean(sign(ŷ) == sign(y))` | Acima de ~52% é economicamente significativo |

R² baixo é esperado e não invalida o modelo: em finanças, até R² de 1–5% pode ser explorado comercialmente. O que importa é se o sinal direcional é suficientemente consistente para gerar alpha após custos.

### Financeiras — qualidade do sinal de trading

| Métrica | Interpretação |
|:--------|:--------------|
| Sharpe anualizado | `(mean(r) / std(r)) × √252`. Sharpe > 1,0 é referência na literatura |
| Máximo drawdown | Maior queda acumulada desde o pico até o vale |
| Alpha vs B&H | Diferença de retorno total entre LSTM e buy & hold simples |

---

## Stack

```bash
pip install torch yfinance pandas-ta scikit-learn seaborn matplotlib scipy statsmodels
```

| Biblioteca | Versão | Uso |
|:----------|:------:|:----|
| `torch` | ≥ 2.0 | Definição da LSTM, loop de treino, loss híbrida |
| `yfinance` | ≥ 0.2 | Download de preços históricos e futuros (WTI, Brent, gás natural) |
| `pandas-ta` | any | Indicadores técnicos. O helper `get_col(df, prefix)` detecta nomes de colunas automaticamente — evita `KeyError` por diferenças de versão |
| `scikit-learn` | ≥ 1.3 | `StandardScaler`, `mean_squared_error`, `r2_score`, `mutual_info_regression` |
| `statsmodels` | any | Teste ADF (Augmented Dickey-Fuller) para verificação de estacionariedade |
| `scipy.stats` | any | Correlação de Pearson e análise de distribuições |
| `matplotlib` / `seaborn` | — | Curvas de capital, scatter de previsões, heatmaps |

---

## Notebooks

### [`exploratory_analysis.ipynb`](exploratory_analysis.ipynb)

Etapa de pré-modelagem: justificativa empírica de todas as decisões de feature engineering. Nenhuma feature foi incluída sem evidência quantitativa.

| Seção | Análise | O que prova |
|:-----:|:--------|:-----------|
| 3 | **Teste ADF** | Log-return é estacionário (p < 0,05); preço bruto não é — justifica a transformação |
| 4 | **Cálculo de features com `pandas-ta`** | Organização em 5 grupos funcionais; todos os indicadores calculados sobre o período de treino |
| 5 | **Correlação — candidatas** | Identifica todos os pares com \|r\| > 0,90, revelando as redundâncias a eliminar |
| 5 | **Correlação — selecionadas** | Confirma ausência de redundância no conjunto final de 13 features |
| 6 | **Mutual Information** | Mede o poder preditivo não-linear de cada feature sobre o retorno do dia seguinte |
| 7 | **Correlação cruzada** | Justifica a arquitetura multivariada: 4 ativos correlacionados mas com divergências temporais exploráveis |
| 8 | **Distribuição das features** | Identifica caudas pesadas e assimetrias que podem impactar o treinamento |
| 9 | **Conclusão** | Consolidação das 13 features com justificativa por critério |

### [`energy-lstm.ipynb`](energy-lstm.ipynb)

Pipeline completo de ponta a ponta:

| # | Seção |
|:-:|:------|
| 1 | Imports e configuração |
| 2 | Coleta de dados via `yfinance` |
| 3 | Engenharia de features com `pandas-ta` |
| 4 | Pipeline — janelas deslizantes, normalização, split |
| 5 | Arquitetura `EnergyLSTM` |
| 6 | Loop de treino com early stopping e gradient clipping |
| 7 | Threshold dinâmico — calibração por percentil no treino |
| 8 | Avaliação estatística no teste |
| 9 | Estratégia de trading com threshold |
| 10 | Comparação com buy & hold — curvas de capital e drawdown |
| 11 | Análise final por ativo e carteira |
| 12 | Simulação de $1.000 investidos |
| 13 | Análise de robustez — 50 seeds com CSV incremental |

---

## Reprodutibilidade

O CSV com os resultados dos 50 seeds é salvo após **cada run individualmente** em `robustez_runs.csv`. Se a execução for interrompida, o notebook detecta o arquivo existente e retoma do seed seguinte automaticamente.

```
[ 1/50]  seed= 0  sharpe=-0.588  alpha=$-146   capital=$776    epochs=100  51s  ✗
[ 2/50]  seed= 1  sharpe=+0.405  alpha=$+344   capital=$1,266  epochs=100  51s  ✓
[11/50]  seed=10  sharpe=+1.249  alpha=$+1,061 capital=$1,983  epochs=100  51s  ✓
...
[50/50]  seed=49  sharpe=+0.522  alpha=$+508   capital=$1,430  epochs=100  51s  ✓
```

Para reproduzir os resultados com `seed=42`:

```bash
jupyter nbconvert --to notebook --execute exploratory_analysis.ipynb --output exploratory_analysis_executed.ipynb
jupyter nbconvert --to notebook --execute energy-lstm.ipynb --output energy-lstm_executed.ipynb
```

---

## Limitações

| Limitação | Impacto |
|:----------|:--------|
| **Custos de transação não modelados** | Bid-ask spread (~0,05%) e comissões reduziriam o alpha realizado. Uma estratégia ativa diária acumula custos relevantes |
| **Short selling sem custo de aluguel** | A estratégia assume posições vendidas sem modelar o custo de aluguel de ações |
| **Sem conjunto de validação independente** | Early stopping monitora loss de treino; ausência de validação reduz sua eficácia como regularizador |
| **Risco de data snooping no threshold** | A seleção do percentil ótimo no treino introduz grau de otimização; validação independente seria mais conservadora |
| **XOM com alpha negativo** | Hipótese: crack spread (margem de refino) é um driver relevante para integradas e não está incluído como feature |
| **Período de teste** | 2023–2025 coincide com recuperação pós-pandemia de SLB/HAL — walk-forward em múltiplos períodos fortaleceria a análise |
| **Benchmark simples** | Buy & hold puro sem comparação com XLE ou estratégias de momentum de referência |

---

## Trabalho Futuro

- [ ] Adicionar **crack spread** (WTI − gasolina) como feature macro para as integradas
- [ ] Incorporar **conjunto de validação independente** (ex.: 2017–2019) para early stopping rigoroso
- [ ] Incorporar **sentiment de notícias** via NLP em headlines do setor de energia
- [ ] Comparar com arquiteturas **Temporal Fusion Transformer (TFT)** e **N-BEATS**
- [ ] Implementar **walk-forward validation** com janelas deslizantes de treino/teste
- [ ] Modelar **custos de transação** explicitamente na função de perda e na simulação
- [ ] Comparar com benchmark setorial **XLE** e estratégias de momentum simples

---

## Referências

- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
- Fischer, T. & Krauss, C. (2018). *Deep learning with long short-term memory networks for financial market predictions*. European Journal of Operational Research, 270(2), 654–669.
- Gu, S., Kelly, B. & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. Review of Financial Studies, 33(5), 2223–2273.
- Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance.
- de Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.
