# SmartInvest AI

**AI-powered market research, risk analysis and portfolio intelligence.**

SmartInvest AI is an independent **hobby and research project** exploring how machine learning, quantitative finance, portfolio optimization and AI agents can be combined into a single investment-analysis system.

The project is designed for experimentation, education and research.

> **SmartInvest is not a financial institution, broker, investment adviser, or automated trading service. Nothing produced by SmartInvest should be considered financial advice.**

---

## Overview

SmartInvest combines multiple analytical systems instead of relying on a single prediction model.

The goal is not simply to predict whether an asset will move up or down. SmartInvest attempts to build a broader view of the market by combining:

* market data
* machine-learning predictions
* market regime detection
* macroeconomic signals
* sentiment analysis
* portfolio risk
* scenario analysis
* portfolio optimization
* explainable AI
* backtesting
* AI-agent reasoning

The final system is designed to act more like an **AI investment research platform** than a simple stock predictor.

---

## Core Architecture

```text
Market Data
    │
    ▼
Data Validation & Feature Engineering
    │
    ├── Technical indicators
    ├── Volatility
    ├── Momentum
    ├── Macro signals
    └── Sentiment
    │
    ▼
Market Regime Detection
    │
    ▼
Multi-Model AI Ensemble
    │
    ├── Transformer
    ├── XGBoost
    ├── Random Forest
    └── Other experimental models
    │
    ▼
Risk Intelligence
    │
    ▼
Strategy Engine
    │
    ▼
Portfolio Optimization
    │
    ▼
Explainable AI
    │
    ▼
Research Report / Portfolio Analysis
```

---

## Features

### Market Data Engine

SmartInvest can work with historical financial market data and maintains local caching where supported.

The data layer is designed around:

* OHLCV market data
* local SQLite caching
* validation and data-quality checks
* retry and fallback mechanisms
* support for multiple assets
* offline-first components in newer versions

---

### Market Regime Detection

SmartInvest attempts to identify the environment in which the market is currently operating.

Possible regimes include:

* **Bull**
* **Bear**
* **Sideways**
* **High Volatility**
* **Crisis**

Regime information can then influence risk limits, model interpretation and portfolio construction.

---

### Multi-Model AI

SmartInvest does not depend on a single machine-learning model.

Different versions of the project experiment with combinations of:

* Transformer models
* XGBoost
* Random Forest
* LightGBM
* LSTM
* reinforcement learning

Predictions from multiple models can be combined into an ensemble.

This allows SmartInvest to compare model agreement and estimate confidence instead of treating one model prediction as absolute truth.

---

### Risk Intelligence

Risk management is one of the central components of SmartInvest.

The system can analyze metrics such as:

* volatility
* Value at Risk (VaR)
* Conditional Value at Risk (CVaR)
* maximum drawdown
* Sharpe ratio
* Sortino ratio
* correlation risk
* portfolio concentration

Newer versions also experiment with dynamic risk controls and volatility-adjusted position sizing.

---

### Portfolio Optimization

SmartInvest includes several approaches to portfolio construction and optimization.

Depending on the version, this can include:

* risk-based optimization
* return/risk optimization
* turnover constraints
* Black-Litterman concepts
* Sortino-based optimization
* reinforcement-learning allocation
* position-size limits

The objective is to construct portfolios while explicitly considering risk instead of simply selecting the assets with the highest predicted return.

---

### Backtesting

SmartInvest includes a backtesting environment for evaluating strategies against historical data.

Later versions include more realistic assumptions such as:

* commissions
* slippage
* transaction costs
* portfolio rebalancing
* turnover constraints
* volatility-scaled trailing stop-losses
* high-water-mark tracking

Backtesting is intended for research purposes and does **not** guarantee future performance.

---

### Explainable AI

Predictions without explanations are difficult to evaluate.

SmartInvest therefore includes explainability systems that can provide information such as:

* why an asset received a certain allocation
* which signals affected the decision
* model confidence
* risk factors
* market regime
* model disagreement
* conditions that could change the decision

The long-term goal is for every major SmartInvest decision to be inspectable rather than functioning as a black box.

---

### Multi-Agent Intelligence

Later SmartInvest generations experiment with specialized AI agents.

Examples include:

* **Macro Agent** — analyzes macroeconomic conditions
* **Alpha Agent** — searches for potential market opportunities
* **Sentiment Agent** — evaluates market sentiment
* **Risk Agent** — focuses on portfolio and market risk
* **Portfolio Agent** — evaluates allocation decisions
* **CIO Agent** — synthesizes the final research view

Agents are designed as research components and do not independently execute real-money trades.

---

## SmartInvest Generations

SmartInvest has evolved through multiple experimental generations.

### SmartInvest 3.x

Early development of the production-oriented architecture, validation, model ensembles and market analysis.

### SmartInvest 4.x

Major expansion of the research platform.

Important additions across the 4.x generation include:

* explainable AI
* adaptive model ensembles
* market regime intelligence
* multi-agent systems
* improved risk management
* realistic backtesting
* portfolio optimization
* transaction costs and slippage
* dynamic trailing stop-loss systems

### SmartInvest 5.x

The 5.x generation focuses on a more advanced AI architecture.

Research areas include:

* time-series Transformers
* stronger portfolio intelligence
* reinforcement-learning allocation
* AI CIO synthesis
* improved validation
* stronger production infrastructure
* more reliable fallbacks
* better testing and evaluation

SmartInvest 5.x remains an actively evolving research branch.

---

## Technology

SmartInvest is primarily developed in **Python**.

The project has experimented with libraries including:

```text
NumPy
pandas
SciPy
scikit-learn
XGBoost
LightGBM
TensorFlow
yfinance
hmmlearn
Plotly
Gymnasium
Stable-Baselines3
SQLite
```

Not every SmartInvest version requires every dependency.

---

## Project Philosophy

SmartInvest follows several basic principles.

### 1. Risk before return

A high predicted return is not useful if the associated risk is ignored.

### 2. Multiple models are better than blind confidence in one model

Different models fail in different ways.

### 3. Market conditions matter

A strategy that performs well in a bull market may behave very differently during a crisis.

### 4. Decisions should be explainable

The system should be able to provide a reason for important recommendations and portfolio changes.

### 5. Backtests must become increasingly realistic

Transaction costs, slippage, turnover and data leakage can completely change apparent performance.

### 6. Uncertainty should be visible

SmartInvest should be able to return **HOLD**, low confidence or insufficient-data states instead of always generating a trade signal.

---

## Current Status

SmartInvest is an **independent hobby and research project under active development**.

It is not a commercial investment platform and there is currently no public pricing structure associated with the project.

Some experimental SmartInvest models may remain private or available only to selected testers.

---

## Model Access

If you are interested in testing or researching SmartInvest models, access may be provided manually on a limited basis.

Requests can include:

* your name
* email address
* reason for requesting access
* intended use
* relevant background or organization, if applicable

Access is not guaranteed.

---

## Security

API credentials should never be committed directly to the repository.

Use environment variables instead.

Example:

```bash
export FRED_API_KEY="your_key"
export NEWSAPI_KEY="your_key"
export ALPHAVANTAGE_API_KEY="your_key"
```

On Windows PowerShell:

```powershell
$env:FRED_API_KEY="your_key"
$env:NEWSAPI_KEY="your_key"
$env:ALPHAVANTAGE_API_KEY="your_key"
```

Never publish real API keys in commits, issues, screenshots or documentation.

---

## Disclaimer

**SmartInvest AI is experimental software created for educational and research purposes.**

It is:

* not financial advice
* not investment advice
* not a registered investment adviser
* not a broker
* not an exchange
* not an automated trading service
* not a guarantee of future investment performance

Financial markets involve substantial risk.

Model predictions can be inaccurate. Data can be incomplete or incorrect. Backtested results can differ significantly from real-world performance.

**Past performance does not guarantee future results.**

Any investment decisions made using information generated by SmartInvest are the responsibility of the user.

---

## Development

SmartInvest is continuously evolving.

Current research priorities include:

* stronger walk-forward validation
* leakage prevention
* improved confidence calibration
* more robust portfolio optimization
* better model evaluation
* stronger testing infrastructure
* improved market-regime detection
* more reliable data pipelines
* better explainability
* safer AI-agent orchestration

---

## Contributing

SmartInvest is currently primarily a personal research project.

Suggestions, research discussions and technical feedback are welcome.

Before proposing major changes, please open an issue describing:

1. the problem
2. the proposed solution
3. expected benefits
4. potential risks or limitations

---

## Contact

For questions about SmartInvest or requests for model access:

**[smartinvest.models@outlook.com](mailto:smartinvest.models@outlook.com)**

---

<p align="center">
  <strong>SmartInvest AI</strong><br>
  Researching the intersection of AI, quantitative finance and portfolio intelligence.
</p>

