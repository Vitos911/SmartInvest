# SmartInvest
🚀 SmartInvest 4.1
AI-Powered Investment System with Full Decision Explainability
SmartInvest 4.1 is an advanced AI-driven portfolio optimization and decision-support system that combines modern machine learning, market regime detection, sentiment & geopolitical analysis, and Black–Litterman optimization — enhanced with a full Decision Explainability Layer.
This project is designed as a research-grade financial AI engine and educational reference system.

🧠 Key Features
🔮 Advanced AI & Forecasting
Temporal Fusion Transformer (TFT)
LSTM Neural Networks
XGBoost
Random Forest
Adaptive ensemble prediction with uncertainty & confidence scoring
🌍 Market Intelligence
Market regime detection (bull / bear / sideways / chaos)
Sentiment analysis from live news data
Geopolitical risk assessment
Dynamic feature engineering with 30+ technical indicators
💼 Portfolio Optimization
Black–Litterman optimization
Risk-aware allocation with configurable risk profiles
Volatility & drawdown controls
Covariance-based risk modeling
🧩 Decision Explainability Layer (SmartInvest 4.1)
The defining feature of version 4.1:
Full explanation of why each investment decision was made
Breakdown of model influence
Risk analysis with explicit warnings
Asset-level explanations (expected return, confidence, sentiment, geopolitical risk)
Clear triggers for strategy change
Human-readable, audit-friendly reporting
This transforms SmartInvest from a black-box AI model into a transparent, professional decision system.
📊 Visualization & Reporting
Interactive Plotly dashboards
Portfolio performance & risk visualization
Structured console reports & AI explanations

🏗️ System Architecture
Market Data
   ↓
Feature Engineering
   ↓
AI Models & Ensemble
   ↓
Regime Detection + Sentiment + Geo Analysis
   ↓
Black–Litterman Optimization
   ↓
Decision Explainability Engine
   ↓
Reports & Visual Dashboard


🛠️ Installation
pip install numpy pandas yfinance scikit-learn scipy plotly tensorflow xgboost lightgbm

TensorFlow, XGBoost and LightGBM are optional — SmartInvest automatically falls back if unavailable.

▶️ Usage
python SmartInvest_4_1.py

Follow the on-screen instructions:
Accept the disclaimer
Enter your capital
Select risk profile (Conservative / Moderate / Aggressive)
The system will:
Fetch market data
Train AI models
Generate predictions
Optimize the portfolio
Produce full explainability & visual reports

📂 Project Highlights
Production-grade logging & caching
SQLite data storage
Modular, extensible design
Research-ready codebase

⚠️ Disclaimer
This software is for educational and research purposes only.
 It is NOT financial advice.
 The authors take no responsibility for financial losses.

🌟 Why SmartInvest 4.1?
Most financial AI systems provide answers.
 SmartInvest 4.1 provides understanding.
By integrating a complete Decision Explainability Layer, SmartInvest becomes:
transparent
auditable
trustworthy
suitable for real-world financial research
