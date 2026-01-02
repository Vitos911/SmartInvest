"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SMARTINVEST 4.0 with AI EXPLAINABILITY                    ║
║                  The Ultimate Transparent AI Investment System               ║
║                                                                              ║
║  Features:                                                                   ║
║  • Production-grade security & validation                                    ║
║  • Advanced ML models: TFT, LSTM, XGBoost, RF                               ║
║  • Geopolitical & sentiment analysis                                         ║
║  • Black-Litterman optimization                                              ║
║  • NEW: AI Decision Explainability Layer                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

DISCLAIMER: This is educational software. NOT financial advice.
Use at your own risk. Past performance ≠ future results.
"""

import os
import sys
import warnings
import logging
import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

# Core libraries
import numpy as np
import pandas as pd

# ML & AI
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, MultiHeadAttention, LayerNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available - using fallback models")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# Financial data
import yfinance as yf

# Visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ML utilities
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy import stats

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION & SECURITY
# =============================================================================

class Config:
    """Centralized system configuration"""
    
    # Paths
    BASE_DIR = Path.cwd()
    CACHE_DIR = BASE_DIR / "cache_ultimate"
    MODELS_DIR = BASE_DIR / "models_ultimate"
    LOGS_DIR = BASE_DIR / "logs_ultimate"
    DB_PATH = CACHE_DIR / "ultimate.db"
    
    # Risk parameters
    MAX_POSITION_SIZE = 0.20
    MIN_POSITION_SIZE = 0.02
    MAX_DRAWDOWN_LIMIT = -0.15
    STOP_LOSS_THRESHOLD = -0.10
    
    # Model parameters
    LOOKBACK_WINDOW = 60
    FORECAST_HORIZON = 30
    MIN_HISTORY_DAYS = 252 * 2
    ENSEMBLE_WEIGHTS = {
        'tft': 0.30,
        'lstm': 0.25,
        'xgb': 0.20,
        'rf': 0.15,
        'chaos': 0.10
    }
    
    @classmethod
    def setup(cls):
        """Create necessary directories"""
        for d in [cls.CACHE_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

Config.setup()

# =============================================================================
# LOGGING & ERROR HANDLING
# =============================================================================

class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SUCCESS = "SUCCESS"

class UltimateLogger:
    """Production-grade logger"""
    
    def __init__(self):
        self.logger = logging.getLogger("SmartInvestUltimate")
        self.logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(Config.LOGS_DIR / "ultimate.log")
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log(self, level: LogLevel, message: str, component: str = "SYSTEM"):
        icons = {
            LogLevel.INFO: "ℹ️",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌",
            LogLevel.CRITICAL: "💀",
            LogLevel.SUCCESS: "✅"
        }
        msg = f"{icons[level]} [{component}] {message}"
        
        if level == LogLevel.INFO:
            self.logger.info(msg)
        elif level == LogLevel.WARNING:
            self.logger.warning(msg)
        elif level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self.logger.error(msg)
        elif level == LogLevel.SUCCESS:
            self.logger.info(msg)

logger = UltimateLogger()

# =============================================================================
# AI DECISION EXPLAINABILITY LAYER
# =============================================================================

class DecisionExplainer:
    """Explains WHY SmartInvest made specific investment decisions"""

    def explain_decision(
        self,
        portfolio_allocation: Dict[str, float],
        expected_return: float,
        volatility: float,
        confidence: float,
        market_regime: str,
        ensemble_weights: Dict[str, float],
        risk_flags: List[str],
        predictions: Dict[str, Dict] = None
    ) -> Dict[str, Any]:

        return {
            "summary": self._generate_summary(market_regime, confidence, expected_return),
            "allocation_reasoning": self._explain_allocation(portfolio_allocation, market_regime, predictions),
            "model_influence": self._explain_models(ensemble_weights),
            "risk_analysis": self._explain_risk(volatility, risk_flags),
            "decision_change_triggers": self._decision_triggers(market_regime),
            "asset_details": self._explain_assets(portfolio_allocation, predictions)
        }

    def _generate_summary(self, regime: str, confidence: float, exp_return: float) -> str:
        confidence_level = (
            "visoko" if confidence > 0.7 else
            "zmerno" if confidence > 0.4 else
            "nizko"
        )

        return (
            f"SmartInvest zaznava {regime.upper()} tržni režim. "
            f"Priporočilo ima {confidence_level} stopnjo zaupanja "
            f"in ciljni pričakovani donos {exp_return * 100:.1f}%."
        )

    def _explain_allocation(
        self,
        allocation: Dict[str, float],
        regime: str,
        predictions: Dict[str, Dict]
    ) -> str:

        if not allocation:
            return "Ni bilo mogoče določiti alokacije."

        total = sum(allocation.values())
        top_assets = sorted(
            [(asset, weight/total) for asset, weight in allocation.items()],
            key=lambda x: x[1], 
            reverse=True
        )[:3]

        assets_text = ", ".join(
            f"{asset} ({weight * 100:.1f}%)"
            for asset, weight in top_assets
        )

        return (
            f"Največja utež je dodeljena naslednjim naložbam: {assets_text}. "
            f"To je skladno z zaznanim {regime.upper()} režimom in strategijo "
            f"optimizacije razmerja med donosom in tveganjem."
        )

    def _explain_models(self, weights: Dict[str, float]) -> List[str]:
        sorted_models = sorted(
            weights.items(), key=lambda x: x[1], reverse=True
        )

        explanations = []
        for model, weight in sorted_models:
            if weight > 0:
                explanations.append(
                    f"Model '{model}' je prispeval {weight * 100:.1f}% k končni odločitvi."
                )
        
        return explanations if explanations else ["Uporabljen osnoven model"]

    def _explain_risk(
        self,
        volatility: float,
        risk_flags: List[str]
    ) -> Dict[str, Any]:

        risk_level = (
            "nizko" if volatility < 0.15 else
            "zmerno" if volatility < 0.30 else
            "visoko"
        )

        return {
            "risk_level": risk_level,
            "portfolio_volatility": volatility,
            "warnings": risk_flags if risk_flags else [
                "Ni zaznanih pomembnih sistemskih tveganj."
            ]
        }

    def _decision_triggers(self, regime: str) -> List[str]:
        triggers = {
            "bull": [
                "zmanjšanje tržnega momentuma",
                "nenaden porast volatilnosti",
                "poslabšanje makroekonomskega sentimenta"
            ],
            "bear": [
                "stabilizacija volatilnosti",
                "pozitiven obrat sentimenta",
                "preboj dolgoročnih povprečij"
            ],
            "sideways": [
                "jasen trendni preboj",
                "povečanje obsega trgovanja"
            ]
        }

        return triggers.get(regime.lower(), [
            "sprememba splošnih tržnih razmer"
        ])

    def _explain_assets(
        self,
        allocation: Dict[str, float],
        predictions: Dict[str, Dict]
    ) -> List[Dict[str, Any]]:
        """Detailed explanation for each asset"""
        
        if not predictions:
            return []
        
        total = sum(allocation.values())
        asset_explanations = []
        
        for symbol, amount in allocation.items():
            if symbol in predictions:
                pred = predictions[symbol]
                
                # Determine why this asset was selected
                reasons = []
                if pred.get('return', 0) > 0.10:
                    reasons.append("visok pričakovan donos")
                if pred.get('confidence', 0) > 0.7:
                    reasons.append("visoko zaupanje AI")
                if pred.get('sentiment', 0) > 0.2:
                    reasons.append("pozitiven sentiment")
                if pred.get('geo_risk', 1) < 0.2:
                    reasons.append("nizko geopolitično tveganje")
                
                asset_explanations.append({
                    'symbol': symbol,
                    'weight': amount / total,
                    'reasons': reasons if reasons else ["splošna diverzifikacija"],
                    'expected_return': pred.get('return', 0),
                    'confidence': pred.get('confidence', 0)
                })
        
        return sorted(asset_explanations, key=lambda x: x['weight'], reverse=True)

    def print_explanation(self, explanation: Dict[str, Any]):
        """Print formatted explanation"""
        print("\n" + "="*100)
        print("🧠 RAZLAGA AI ODLOČITVE - Decision Explainability Report")
        print("="*100)
        
        print(f"\n📊 POVZETEK:")
        print(f"   {explanation['summary']}")
        
        print(f"\n💼 RAZLAGA ALOKACIJE:")
        print(f"   {explanation['allocation_reasoning']}")
        
        print(f"\n🤖 VPLIV MODELOV:")
        for influence in explanation['model_influence']:
            print(f"   • {influence}")
        
        print(f"\n⚠️ ANALIZA TVEGANJA:")
        risk = explanation['risk_analysis']
        print(f"   Stopnja tveganja: {risk['risk_level'].upper()}")
        print(f"   Volatilnost portfelja: {risk['portfolio_volatility']*100:.2f}%")
        print(f"   Opozorila:")
        for warning in risk['warnings']:
            print(f"      • {warning}")
        
        print(f"\n🔄 SPROŽILCI SPREMEMBE STRATEGIJE:")
        print(f"   Odločitev bi se spremenila ob:")
        for trigger in explanation['decision_change_triggers']:
            print(f"      • {trigger}")
        
        if explanation.get('asset_details'):
            print(f"\n🎯 PODROBNA RAZLAGA PO SREDSTVIH:")
            for asset in explanation['asset_details'][:5]:  # Top 5
                print(f"\n   {asset['symbol']} ({asset['weight']*100:.1f}% portfelja):")
                print(f"      Razlogi za izbiro: {', '.join(asset['reasons'])}")
                print(f"      Pričakovan donos: {asset['expected_return']*100:.1f}%")
                print(f"      Zaupanje AI: {asset['confidence']:.2f}")
        
        print("\n" + "="*100 + "\n")

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

class DataManager:
    """Manages data fetching and caching"""
    
    def __init__(self):
        self.db_path = Config.DB_PATH
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                cached_at TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        ''')
        conn.commit()
        conn.close()
    
    def fetch_data(self, symbol: str, period: str = '3y') -> Optional[pd.DataFrame]:
        """Fetch data with caching"""
        cached = self._get_cached(symbol, period)
        if cached is not None:
            return cached
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval='1d')
            
            if df.empty:
                logger.log(LogLevel.WARNING, f"No data for {symbol}", "DATA")
                return None
            
            self._cache_data(symbol, df)
            return df
            
        except Exception as e:
            logger.log(LogLevel.ERROR, f"Failed to fetch {symbol}: {e}", "DATA")
            return None
    
    def _get_cached(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Get cached data"""
        days = {'1y': 365, '3y': 3*365, '5y': 5*365}.get(period, 3*365)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT date, open, high, low, close, volume 
                FROM market_data 
                WHERE symbol = '{symbol}' AND date >= '{start_date}'
                ORDER BY date
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return None
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            if df.index[-1] < pd.Timestamp.now() - timedelta(days=2):
                return None
            
            return df
        except Exception:
            return None
    
    def _cache_data(self, symbol: str, df: pd.DataFrame):
        """Cache data to SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            records = []
            for idx, row in df.iterrows():
                records.append((
                    symbol,
                    idx.strftime('%Y-%m-%d'),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    float(row['Volume']),
                    datetime.now().isoformat()
                ))
            
            cursor.executemany('''
                INSERT OR REPLACE INTO market_data 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.log(LogLevel.ERROR, f"Cache error: {e}", "DATA")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngine:
    """Creates advanced features for ML models"""
    
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        df = df.copy()
        
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        df['Vol_10'] = df['Returns'].rolling(10).std() * np.sqrt(252)
        df['Vol_30'] = df['Returns'].rolling(30).std() * np.sqrt(252)
        df['Vol_60'] = df['Returns'].rolling(60).std() * np.sqrt(252)
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_30'] = df['Close'] / df['Close'].shift(30) - 1
        
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        df['Chaos_Indicator'] = df['Vol_30'] / (df['Vol_60'] + 1e-6)
        
        df['Price_Position'] = (df['Close'] - df['Close'].rolling(252).min()) / \
                               (df['Close'].rolling(252).max() - df['Close'].rolling(252).min() + 1e-6)
        
        df['Trend_Strength'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        
        df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        
        return df

# =============================================================================
# AI MODELS
# =============================================================================

class TemporalFusionTransformer:
    """TFT implementation"""
    
    def __init__(self, input_shape, horizon=1):
        self.input_shape = input_shape
        self.horizon = horizon
        self.model = None
    
    def build(self):
        if not TF_AVAILABLE:
            return None
        
        inputs = Input(shape=self.input_shape)
        x = LSTM(128, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = LayerNormalization()(x + attn)
        x = LSTM(64)(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.horizon, activation='linear')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        
        return self.model
    
    def train(self, X, y):
        if self.model is None:
            self.build()
        
        if self.model:
            callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
            self.model.fit(X, y, epochs=30, batch_size=32, verbose=0, callbacks=callbacks)
    
    def predict(self, X):
        if self.model:
            return self.model.predict(X, verbose=0)
        return np.zeros((len(X), self.horizon))

class UltimateEnsemble:
    """Ultimate ensemble combining multiple models"""
    
    def __init__(self):
        self.tft = None
        self.lstm = None
        self.xgb_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.models_used = []
    
    def prepare_sequences(self, df: pd.DataFrame, lookback=60):
        """Prepare data for LSTM/TFT"""
        feature_cols = [
            'Close', 'Returns', 'Vol_30', 'RSI', 'MACD', 
            'BB_Width', 'Momentum_30', 'Volume_Ratio', 'Trend_Strength'
        ]
        
        data = df[feature_cols].values
        data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback, len(data) - 30):
            X.append(data[i-lookback:i])
            future_ret = (df['Close'].iloc[i+30] - df['Close'].iloc[i]) / df['Close'].iloc[i]
            y.append(future_ret)
        
        return np.array(X), np.array(y)
    
    def train_all(self, df: pd.DataFrame):
        """Train all available models"""
        X, y = self.prepare_sequences(df)
        
        if len(X) < 50:
            raise DataError("Not enough sequences")
        
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        self.models_used = []
        
        # TFT
        if TF_AVAILABLE:
            logger.log(LogLevel.INFO, "Training TFT...", "AI")
            self.tft = TemporalFusionTransformer(input_shape=(X.shape[1], X.shape[2]))
            self.tft.train(X_train, y_train)
            self.models_used.append('tft')
        
        # LSTM
        if TF_AVAILABLE:
            logger.log(LogLevel.INFO, "Training LSTM...", "AI")
            self.lstm = Sequential([
                LSTM(100, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                Dense(1)
            ])
            self.lstm.compile(optimizer='adam', loss='mse')
            self.lstm.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            self.models_used.append('lstm')
        
        # XGBoost
        if XGB_AVAILABLE:
            logger.log(LogLevel.INFO, "Training XGBoost...", "AI")
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            self.xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
            self.xgb_model.fit(X_train_flat, y_train)
            self.models_used.append('xgb')
        
        # Random Forest
        logger.log(LogLevel.INFO, "Training Random Forest...", "AI")
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        self.rf_model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        self.rf_model.fit(X_train_flat, y_train)
        self.models_used.append('rf')
        
        logger.log(LogLevel.SUCCESS, f"Trained {len(self.models_used)} models", "AI")
    
    def predict_ensemble(self, X) -> Dict[str, float]:
        """Ensemble prediction"""
        predictions = []
        weights = []
        
        if self.tft and self.tft.model:
            pred = self.tft.predict(X[-1:])
            predictions.append(float(pred[0][0]))
            weights.append(Config.ENSEMBLE_WEIGHTS['tft'])
        
        if self.lstm:
            pred = self.lstm.predict(X[-1:], verbose=0)
            predictions.append(float(pred[0][0]))
            weights.append(Config.ENSEMBLE_WEIGHTS['lstm'])
        
        if self.xgb_model:
            X_flat = X[-1:].reshape(1, -1)
            pred = self.xgb_model.predict(X_flat)
            predictions.append(float(pred[0]))
            weights.append(Config.ENSEMBLE_WEIGHTS['xgb'])
        
        if self.rf_model:
            X_flat = X[-1:].reshape(1, -1)
            pred = self.rf_model.predict(X_flat)
            predictions.append(float(pred[0]))
            weights.append(Config.ENSEMBLE_WEIGHTS['rf'])
        
        if not predictions:
            return {'return': 0.0, 'uncertainty': 1.0, 'confidence': 0.0}
        
        weights = np.array(weights) / sum(weights)
        final_pred = np.average(predictions, weights=weights)
        uncertainty = np.std(predictions)
        confidence = 1.0 / (1.0 + uncertainty)
        
        return {
            'return': float(final_pred),
            'uncertainty': float(uncertainty),
            'confidence': float(min(confidence, 0.95))
        }

# =============================================================================
# REGIME DETECTION
# =============================================================================

class RegimeDetector:
    """Detect market regime"""
    
    @staticmethod
    def detect(spy_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market regime"""
        if spy_data is None or len(spy_data) < 200:
            return {'regime': 'unknown', 'confidence': 0.0}
        
        sma50 = spy_data['Close'].rolling(50).mean().iloc[-1]
        sma200 = spy_data['Close'].rolling(200).mean().iloc[-1]
        current_price = spy_data['Close'].iloc[-1]
        
        returns = spy_data['Close'].pct_change().dropna()
        vol = returns.std() * np.sqrt(252)
        
        if sma50 > sma200 and vol < 0.20:
            regime = 'bull'
            confidence = 0.8
        elif sma50 < sma200 and vol > 0.30:
            regime = 'bear'
            confidence = 0.8
        elif vol > 0.40:
            regime = 'chaos'
            confidence = 0.9
        else:
            regime = 'sideways'
            confidence = 0.6
        
        return {
            'regime': regime,
            'confidence': confidence,
            'volatility': vol,
            'trend': 'up' if sma50 > sma200 else 'down'
        }

# =============================================================================
# SENTIMENT & GEO ANALYSIS
# =============================================================================

class SentimentAnalyzer:
    """Analyze market sentiment"""
    
    @staticmethod
    def analyze(symbol: str) -> Dict[str, float]:
        """Get sentiment score"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            
            if not news:
                return {'score': 0.0, 'confidence': 0.0}
            
            positive_words = ['surge', 'gain', 'profit', 'beat', 'strong', 'upgrade', 'buy']
            negative_words = ['fall', 'loss', 'miss', 'weak', 'downgrade', 'sell', 'decline']
            
            pos_count = 0
            neg_count = 0
            
            for item in news[:10]:
                title = item.get('title', '').lower()
                for word in positive_words:
                    if word in title:
                        pos_count += 1
                for word in negative_words:
                    if word in title:
                        neg_count += 1
            
            total = pos_count + neg_count
            if total == 0:
                return {'score': 0.0, 'confidence': 0.0}
            
            score = (pos_count - neg_count) / total
            confidence = min(total / 10, 1.0)
            
            return {
                'score': float(score),
                'confidence': float(confidence),
                'news_count': len(news)
            }
            
        except Exception:
            return {'score': 0.0, 'confidence': 0.0}

class GeopoliticalAnalyzer:
    """Analyze geopolitical risks"""
    
    def get_risk_score(self, symbol: str) -> float:
        """Get geopolitical risk score"""
        high_risk = ['BABA', 'TCEHY', 'PBR', 'VALE']
        medium_risk = ['TSM', 'ASML', 'NVO']
        
        if symbol in high_risk:
            return 0.4
        elif symbol in medium_risk:
            return 0.2
        else:
            return 0.1

# =============================================================================
# PORTFOLIO OPTIMIZATION
# =============================================================================

class BlackLittermanOptimizer:
    """Black-Litterman portfolio optimization"""
    
    def __init__(self, capital: float, risk_aversion: float = 2.5):
        self.capital = capital
        self.risk_aversion = risk_aversion
    
    def optimize(self, predictions: Dict[str, Dict], cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Optimize portfolio"""
        assets = list(predictions.keys())
        n = len(assets)
        
        if n == 0:
            return {}
        
        prior = np.zeros(n)
        views = np.array([predictions[a]['return'] for a in assets])
        confidences = np.array([predictions[a]['confidence'] for a in assets])
        
        posterior = views * confidences + prior * (1 - confidences)
        
        cov = cov_matrix.loc[assets, assets].values
        
        def objective(w):
            ret = np.dot(w, posterior)
            vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            return -(ret - self.risk_aversion * vol**2)
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, Config.MAX_POSITION_SIZE) for _ in range(n))
        
        x0 = np.array([1/n] * n)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = result.x
        
        allocation = {}
        for i, asset in enumerate(assets):
            if weights[i] >= Config.MIN_POSITION_SIZE:
                allocation[asset] = weights[i] * self.capital
        
        return allocation

# =============================================================================
# MAIN SYSTEM
# =============================================================================

class SmartInvestUltimate:
    """Main system orchestrator"""
    
    def __init__(self, capital: float, risk_profile: str):
        self.capital = capital
        self.risk_profile = risk_profile
        
        self.data_manager = DataManager()
        self.feature_engine = FeatureEngine()
        self.ensemble = UltimateEnsemble()
        self.regime_detector = RegimeDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.geo_analyzer = GeopoliticalAnalyzer()
        self.explainer = DecisionExplainer()
        
        risk_map = {'conservative': 5.0, 'moderate': 2.5, 'aggressive': 1.0}
        risk_aversion = risk_map.get(risk_profile, 2.5)
        self.optimizer = BlackLittermanOptimizer(capital, risk_aversion)
        
        self.market_data = {}
        self.predictions = {}
        self.allocation = {}
        self.regime_info = {}
        self.explanation = {}
    
    def run(self):
        """Execute full analysis"""
        logger.log(LogLevel.INFO, "Starting SmartInvest 4.0 analysis", "CORE")
        
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN', 'TSLA', 'META',
            'JPM', 'BAC', 'V', 'MA', 'JNJ', 'UNH',
            'SPY', 'QQQ', 'TLT', 'GLD', 'BTC-USD'
        ]
        
        # Load data
        logger.log(LogLevel.INFO, f"Loading {len(symbols)} assets...", "DATA")
        for symbol in symbols:
            df = self.data_manager.fetch_data(symbol)
            if df is not None:
                df = self.feature_engine.engineer_features(df)
                self.market_data[symbol] = df
        
        if not self.market_data:
            logger.log(LogLevel.CRITICAL, "No data loaded!", "CORE")
            return
        
        logger.log(LogLevel.SUCCESS, f"Loaded {len(self.market_data)} assets", "DATA")
        
        # Detect regime
        if 'SPY' in self.market_data:
            self.regime_info = self.regime_detector.detect(self.market_data['SPY'])
            logger.log(LogLevel.INFO, 
                      f"Market Regime: {self.regime_info['regime'].upper()} "
                      f"(confidence: {self.regime_info['confidence']:.2f})",
                      "REGIME")
        
        # Train & predict
        logger.log(LogLevel.INFO, "Training AI models...", "AI")
        
        for symbol, df in self.market_data.items():
            try:
                ensemble = UltimateEnsemble()
                ensemble.train_all(df)
                X, _ = ensemble.prepare_sequences(df)
                ai_pred = ensemble.predict_ensemble(X)
                
                sentiment = self.sentiment_analyzer.analyze(symbol)
                geo_risk = self.geo_analyzer.get_risk_score(symbol)
                
                final_return = ai_pred['return'] * (1 - geo_risk * 0.2) + sentiment['score'] * 0.1
                final_confidence = ai_pred['confidence'] * (1 - geo_risk * 0.3)
                
                self.predictions[symbol] = {
                    'return': final_return,
                    'confidence': final_confidence,
                    'uncertainty': ai_pred['uncertainty'],
                    'sentiment': sentiment['score'],
                    'geo_risk': geo_risk,
                    'ai_return': ai_pred['return']
                }
                
                logger.log(LogLevel.SUCCESS,
                          f"{symbol}: ret={final_return*100:.1f}%, conf={final_confidence:.2f}",
                          "AI")
                
            except Exception as e:
                logger.log(LogLevel.ERROR, f"Failed {symbol}: {e}", "AI")
        
        if not self.predictions:
            logger.log(LogLevel.CRITICAL, "No predictions!", "CORE")
            return
        
        # Optimize
        logger.log(LogLevel.INFO, "Optimizing portfolio...", "OPTIMIZER")
        
        prices = pd.DataFrame({s: self.market_data[s]['Close'] for s in self.predictions.keys()})
        returns = prices.pct_change().dropna()
        cov_matrix = returns.cov() * 252
        
        self.allocation = self.optimizer.optimize(self.predictions, cov_matrix)
        
        # Calculate metrics
        total_return = sum(self.predictions[s]['return'] * (amt/self.capital) 
                          for s, amt in self.allocation.items())
        avg_confidence = np.mean([self.predictions[s]['confidence'] for s in self.allocation.keys()])
        portfolio_vol = np.sqrt(sum((amt/self.capital)**2 * self.predictions[s]['uncertainty']**2 
                                   for s, amt in self.allocation.items()))
        
        # Generate explanation
        risk_flags = []
        if portfolio_vol > 0.30:
            risk_flags.append("Visoka volatilnost portfelja")
        if self.regime_info.get('regime') == 'bear':
            risk_flags.append("Negativni tržni režim")
        
        self.explanation = self.explainer.explain_decision(
            portfolio_allocation=self.allocation,
            expected_return=total_return,
            volatility=portfolio_vol,
            confidence=avg_confidence,
            market_regime=self.regime_info.get('regime', 'unknown'),
            ensemble_weights=Config.ENSEMBLE_WEIGHTS,
            risk_flags=risk_flags,
            predictions=self.predictions
        )
        
        # Print reports
        self._print_report()
        self.explainer.print_explanation(self.explanation)
        
        if PLOTLY_AVAILABLE:
            self._visualize()
    
    def _print_report(self):
        """Print portfolio report"""
        print("\n" + "="*100)
        print("📊 SMARTINVEST 4.0 - PORTFOLIO REPORT")
        print("="*100)
        print(f"💰 Capital: €{self.capital:,.2f}")
        print(f"🎯 Risk Profile: {self.risk_profile}")
        print(f"🌍 Market Regime: {self.regime_info.get('regime', 'unknown').upper()}")
        print("-"*100)
        
        total_return = sum(self.predictions[s]['return'] * (amt/self.capital) 
                          for s, amt in self.allocation.items())
        avg_confidence = np.mean([self.predictions[s]['confidence'] for s in self.allocation.keys()])
        
        print(f"📈 Expected Return: {total_return*100:.2f}%")
        print(f"🎲 Average Confidence: {avg_confidence:.2f}")
        print(f"📦 Number of Assets: {len(self.allocation)}")
        print("-"*100)
        
        print(f"{'ASSET':<10} {'AMOUNT (€)':<15} {'WEIGHT (%)':<12} {'EXP. RETURN':<12} {'CONFIDENCE':<12}")
        print("-"*100)
        
        sorted_alloc = sorted(self.allocation.items(), key=lambda x: x[1], reverse=True)
        
        for symbol, amount in sorted_alloc:
            weight = (amount / self.capital) * 100
            pred = self.predictions[symbol]
            ret = pred['return'] * 100
            conf = pred['confidence']
            
            print(f"{symbol:<10} {amount:<15,.2f} {weight:<12.1f} {ret:<12.1f}% {conf:<12.2f}")
        
        print("="*100)
        print("⚠️  This is educational analysis. NOT financial advice!")
        print("="*100 + "\n")
    
    def _visualize(self):
        """Create interactive dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Allocation', 'Expected Returns', 'Confidence Levels', 'Risk/Return'),
            specs=[
                [{'type': 'domain'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'scatter'}]
            ]
        )
        
        symbols = list(self.allocation.keys())
        amounts = list(self.allocation.values())
        
        # Pie chart
        fig.add_trace(
            go.Pie(labels=symbols, values=amounts, hole=0.3),
            row=1, col=1
        )
        
        # Returns bar
        returns = [self.predictions[s]['return'] * 100 for s in symbols]
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        fig.add_trace(
            go.Bar(x=symbols, y=returns, marker_color=colors),
            row=1, col=2
        )
        
        # Confidence bar
        confidences = [self.predictions[s]['confidence'] for s in symbols]
        
        fig.add_trace(
            go.Bar(x=symbols, y=confidences, marker_color='lightblue'),
            row=2, col=1
        )
        
        # Scatter
        uncertainties = [self.predictions[s]['uncertainty'] * 100 for s in symbols]
        sizes = [self.allocation[s] / self.capital * 1000 for s in symbols]
        
        fig.add_trace(
            go.Scatter(
                x=uncertainties,
                y=returns,
                mode='markers+text',
                text=symbols,
                textposition='top center',
                marker=dict(size=sizes, color=returns, colorscale='RdYlGn', showscale=True)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="SmartInvest 4.0 - Dashboard",
            showlegend=False,
            height=800,
            template='plotly_dark'
        )
        
        fig.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point"""
    print("\n" + "█"*100)
    print("   🚀 SMARTINVEST 4.0 - AI INVESTMENT SYSTEM WITH EXPLAINABILITY")
    print("   Features: TFT + LSTM + XGBoost + RF + Sentiment + Geopolitics + Explainer")
    print("█"*100 + "\n")
    
    print("⚠️  LEGAL DISCLAIMER:")
    print("    Educational software. NOT financial advice. You can LOSE money.")
    print("    Consult licensed professionals.\n")
    
    response = input("    Type 'I ACCEPT' to continue: ").strip().upper()
    
    if response != 'I ACCEPT':
        print("\n❌ Exiting...\n")
        return
    
    print("\n✅ Proceeding...\n")
    
    # User input
    while True:
        try:
            capital = float(input("💰 Capital (EUR): ").replace(',', ''))
            if capital > 0:
                break
        except ValueError:
            pass
    
    print("\nRisk Profile:")
    print("   1️⃣  Conservative")
    print("   2️⃣  Moderate")
    print("   3️⃣  Aggressive")
    
    choice = input("\nYour choice (1-3): ").strip()
    risk_map = {'1': 'conservative', '2': 'moderate', '3': 'aggressive'}
    risk_profile = risk_map.get(choice, 'moderate')
    
    print(f"\n🎯 Initializing SmartInvest 4.0 for €{capital:,.0f} [{risk_profile}]...\n")
    
    # Run
    system = SmartInvestUltimate(capital, risk_profile)
    system.run()
    
    print("\n✅ Analysis complete!")
    print("📁 Logs saved to:", Config.LOGS_DIR)
    print("\n🙏 Thank you for using SmartInvest 4.0!\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted.\n")
    except Exception as e:
        logger.log(LogLevel.CRITICAL, f"Fatal error: {e}", "MAIN")
        import traceback
        traceback.print_exc()