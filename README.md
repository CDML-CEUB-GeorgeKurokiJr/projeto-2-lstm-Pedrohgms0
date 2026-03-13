# Previsão de Ações do Setor de Energia com LSTM

Projeto de Deep Learning para previsão de preços de ações utilizando redes neurais recorrentes (LSTM multivariada). A carteira é composta por quatro empresas do setor de energia listadas na NYSE, com estratégia de features construída a partir de lógica econômica e análise técnica de mercado.

---

## Carteira

| Ticker | Empresa | Subsetor | Papel na carteira |
|--------|---------|----------|-------------------|
| **XOM** | ExxonMobil | Petróleo integrado | Âncora — benchmark global |
| **CVX** | Chevron | Petróleo integrado | Âncora — benchmark americano |
| **SLB** | Schlumberger | Serviços de exploração | Defasagem temporal em relação às integradas |
| **HAL** | Halliburton | Serviços de completação | Defasagem temporal em relação às integradas |

### Lógica econômica da carteira

A carteira representa uma **cadeia de valor vertical do petróleo**. XOM e CVX são produtoras integradas — respondem diretamente ao preço do WTI. SLB e HAL são prestadoras de serviços — crescem quando as integradas aumentam investimentos em exploração, o que ocorre com um ou dois trimestres de defasagem.

Essa estrutura cria assimetrias temporais naturais que uma LSTM multivariada consegue capturar: o preço do petróleo sobe → XOM e CVX lucram → aumentam o capex → contratam SLB e HAL. O modelo aprende essa cadeia de causalidade implicitamente ao treinar os quatro ativos em conjunto.

---

## Dados e Período

```
Treino:     2010 – 2016
Validação:  2017 – 2018
Teste:      2023 – 2024
```

O período 2019–2022 foi **excluído intencionalmente**. A pandemia de COVID-19 gerou um regime de mercado anômalo — o WTI chegou a preços negativos em abril de 2020, as correlações históricas do setor quebraram e houve intervenções governamentais sem precedente. Incluir esse período ensinaria ao modelo um regime que não se repetirá, contaminando a capacidade de generalização.

**Fonte dos dados:** Yahoo Finance via `yfinance`

---

## Features

O conjunto de features foi construído para cobrir **grupos de informação não redundantes**, eliminando indicadores com correlação acima de 0.90 entre si. Preço bruto foi substituído por log-return em todos os casos para garantir estacionariedade da série.

### Preço e Retorno

| Feature | Cálculo | Por que foi escolhida |
|---------|---------|----------------------|
| Log-return close | `ln(close_t / close_t−1)` | Preço bruto não é comparável entre épocas distintas. O log-return normaliza a escala e torna a série estacionária — propriedade exigida por modelos de séries temporais |
| Log-return open | `ln(open_t / close_t−1)` | Captura surpresas ocorridas fora do pregão (notícias, geopolítica, resultados). Para energia, choques noturnos no WTI se manifestam primeiro no gap de abertura |

### Volume

| Feature | Cálculo | Por que foi escolhida |
|---------|---------|----------------------|
| Volume ratio | `vol_t / média_vol_20d` | Volume bruto não é comparável entre ativos nem entre épocas. O ratio detecta anomalias de participação — volume 2× acima da média sinaliza evento relevante independente do nível absoluto |

### Tendência

| Feature | Cálculo | Por que foi escolhida |
|---------|---------|----------------------|
| SMA diff | `(SMA50 − SMA200) / SMA200` | A SMA200 é seguida por gestores institucionais como fronteira bull/bear market. A diferença percentual é uma profecia autorrealizável com poder preditivo real no setor de energia |
| EMA diff | `(EMA12 − EMA26) / close` | EMA reage mais rápido que SMA por dar peso maior a dados recentes. Complementa a SMA diff — enquanto SMA captura regime de longo prazo, EMA diff captura tendência de curto prazo |

### Momentum

| Feature | Cálculo | Por que foi escolhida |
|---------|---------|----------------------|
| RSI 14 | valor 0–100 | Captura zonas de sobrecompra (>70) e sobrevenda (<30). Já está normalizado entre 0–100. Baixa correlação com MACD — os dois são complementares |
| MACD histograma | normalizado pelo close | RSI mede força, MACD mede aceleração e direção. Quando o histograma muda de sinal antes do preço, antecipa reversões. Normalizar pelo close permite comparação entre ativos e épocas |

### Volatilidade

| Feature | Cálculo | Por que foi escolhida |
|---------|---------|----------------------|
| ATR 14 | `ATR / close` | O setor de energia alterna entre regimes de baixa e alta volatilidade seguindo o ciclo do petróleo. O modelo precisa saber em qual regime está para calibrar suas previsões |
| Bollinger %B | valor 0–1 | Combina tendência e volatilidade em uma métrica só. Acrescenta informação que ATR e RSI isolados não capturam — %B = 1 significa preço na banda superior, %B = 0 na inferior |

### Commodities — Drivers Externos

| Feature | Ticker yfinance | Por que foi escolhida |
|---------|----------------|----------------------|
| WTI log-return | `CL=F` | Driver fundamental direto de XOM, CVX, SLB e HAL. Movimentos do WTI precedem movimentos das ações em horas — feature com poder preditivo causal, não apenas correlacional |
| Brent log-return | `BZ=F` | Captura choques geopolíticos internacionais que movem o Brent antes do WTI. XOM tem operações globais — o Brent reflete o ambiente externo que o WTI ainda não precificou |
| Spread WTI−Brent | `log-return WTI − log-return Brent` | Quando o spread alarga, reflete gargalos de infraestrutura ou desequilíbrios regionais de oferta — informação que nenhum dos dois preços isolados transmite |
| Gás Natural log-return | `NG=F` | XOM e CVX são produtores integrados de gás e petróleo. Os ciclos nem sempre se sincronizam — pós-2022 o mercado de GNL mudou estruturalmente |

---

## Janela de Observação (Lookback)

**Janela escolhida: 20 dias úteis (1 mês de pregão)**

Essa escolha tem três justificativas concretas:

1. **Unidade natural de mercado** — 20 dias é o número de pregões em um mês, que é a unidade de ciclo de decisão de gestores institucionais. Relatórios mensais, rebalanceamentos de carteira e metas de performance são medidos nessa escala.

2. **Cobertura dos indicadores** — o RSI precisa de 14 dias e a SMA20 de 20 dias para produzir valores válidos. Uma janela menor entregaria features mal calculadas no início de cada sequência.

3. **Relevância da informação** — informação com mais de 20 pregões já foi amplamente precificada pelo mercado. Aumentar a janela eleva a dimensão do tensor de entrada sem ganho preditivo proporcional, aumentando o risco de overfitting.

| Alternativa | Problema |
|-------------|----------|
| 5–10 dias | Janela menor que o período dos indicadores — RSI e SMA20 não aqueceram |
| 60–120 dias | Informação antiga já precificada — overfitting sem ganho preditivo |

---

## Arquitetura do Modelo

```
Tensor de entrada: (amostras, 20 passos, 13 features)
```

```python
class EnergyLSTM(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, output_size=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

A LSTM é treinada de forma **multivariada**: os 4 ativos são processados em conjunto como vetor de saída (`output_size=4`), permitindo que o modelo aprenda as relações cruzadas entre eles — especialmente a defasagem temporal entre as integradas (XOM, CVX) e os prestadores de serviços (SLB, HAL).

O pré-processamento (normalização e construção das janelas) é feito com `scikit-learn` (`MinMaxScaler` ou `StandardScaler`) antes de alimentar o modelo PyTorch.

### Função de Perda Híbrida

```
Loss = RMSE + λ · (1 − Sharpe normalizado)
```

A função de perda combina precisão preditiva (RMSE) com performance financeira (Sharpe Ratio). O hiperparâmetro λ controla o equilíbrio entre os dois objetivos — um modelo treinado apenas com RMSE minimiza erro de previsão mas pode gerar sinais de trading ruins.

---

## Split Temporal

> **Importante:** dados financeiros nunca devem ser embaralhados antes do split. O vazamento de dados futuros no treino (*data leakage*) produz métricas artificialmente boas que não se reproduzem em produção.

```
Treino    (70%)  →  ~1.910 amostras  —  2010 a 2016
Validação (15%)  →  ~410 amostras   —  2017 a 2018
Teste     (15%)  →  ~410 amostras   —  2023 a 2024
```

---

## Bibliotecas

| Biblioteca | Uso |
|-----------|-----|
| `torch` | Definição e treino da LSTM |
| `scikit-learn` | Pré-processamento, normalização e métricas |
| `yfinance` | Coleta de dados históricos |
| `pandas-ta` | Cálculo dos indicadores técnicos |
| `pandas` / `numpy` | Manipulação de dados e engenharia de features |
| `matplotlib` / `seaborn` | Visualização e análise exploratória |

---

## Métricas de Avaliação

| Métrica | O que mede |
|---------|-----------|
| RMSE | Erro quadrático médio da previsão de preço |
| MAE | Erro absoluto médio — menos sensível a outliers |
| Acurácia direcional | % de dias em que o modelo acertou a direção (alta/baixa) |
| Sharpe Ratio | Retorno ajustado ao risco da estratégia gerada pelo modelo |


