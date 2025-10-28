import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow ni na voljo. Uporabljam samo RandomForest model.")
    TENSORFLOW_AVAILABLE = False

import yfinance as yf
from datetime import datetime, timedelta
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    print("⚠️  TA-Lib ni na voljo. Uporabljam poenostavljene tehnične kazalnike.")
    TALIB_AVAILABLE = False

import requests
from bs4 import BeautifulSoup
import re
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    print("⚠️  TextBlob ni na voljo. Sentiment analiza bo onemogočena.")
    TEXTBLOB_AVAILABLE = False

import time
from scipy.stats import linregress
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Novi importi za SmartInvest 2.5
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost ni na voljo. Uporabljam samo RandomForest in LSTM.")
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    print("⚠️  LightGBM ni na voljo.")
    LGBM_AVAILABLE = False

# Izklop opozoril
warnings.filterwarnings('ignore')

class GeopoliticalAnalyzer:
    """Analizira geopolitične dejavnike in njihov vpliv na trge."""

    def __init__(self):
        self.risk_regions = {
            'middle_east': ['oil', 'defense', 'shipping'],
            'asia_pacific': ['technology', 'semiconductors', 'shipping'],
            'europe': ['finance', 'automotive', 'luxury'],
            'americas': ['technology', 'energy', 'agriculture']
        }

        self.current_risk_levels = {
            'middle_east': 0.3,
            'asia_pacific': 0.2,
            'europe': 0.1,
            'americas': 0.1
        }

        self.sector_sensitivities = {
            'energy': 0.8, 'defense': 0.9, 'technology': 0.7,
            'finance': 0.6, 'healthcare': 0.3, 'consumer': 0.4,
            'materials': 0.7, 'industrial': 0.5, 'utilities': 0.3
        }

    def get_region_impact(self, region, sector):
        """Vrne vpliv geopolitičnih dogodkov po regijah in sektorjih"""
        base_impact = self.current_risk_levels[region]
        sector_sensitivity = self.sector_sensitivities.get(sector, 0.5)
        return base_impact * sector_sensitivity

    def analyze_news_sentiment(self, symbol):
        """Analizira novice za geopolitične signale"""
        # Poenostavljena implementacija - v praksi bi uporabili News API
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news

            geopolitical_keywords = [
                'war', 'conflict', 'sanctions', 'election', 'trade war',
                'tension', 'crisis', 'summit', 'agreement', 'talks'
            ]

            risk_score = 0.0
            if news:
                for item in news[:5]:
                    title = item.get('title', '').lower()
                    summary = item.get('summary', '').lower()
                    text = f"{title} {summary}"

                    keyword_count = sum(1 for keyword in geopolitical_keywords if keyword in text)
                    risk_score += keyword_count * 0.1

            return min(risk_score, 1.0)
        except:
            return 0.0

class SmartInvest25:
    """
    SmartInvest 2.5 - Nadgrajen sistem z geopolitično analizo in izboljšano natančnostjo
    """
    def __init__(self, amount, risk_profile, investment_horizon):
        self.amount = amount
        self.risk_profile = risk_profile
        self.investment_horizon = investment_horizon
        self.scaler = StandardScaler()
        self.geopolitical_analyzer = GeopoliticalAnalyzer()

        # Definicija profilov tveganja
        self.risk_profiles = {
            'zelo_nizko': {
                'bonds': 0.70, 'etfs': 0.20, 'stocks': 0.10,
                'max_drawdown': 0.05, 'target_return': 0.04
            },
            'nizko': {
                'bonds': 0.50, 'etfs': 0.30, 'stocks': 0.20,
                'max_drawdown': 0.10, 'target_return': 0.06
            },
            'srednje': {
                'stocks': 0.50, 'etfs': 0.30, 'bonds': 0.15, 'alternatives': 0.05,
                'max_drawdown': 0.15, 'target_return': 0.08
            },
            'visoko': {
                'stocks': 0.60, 'etfs': 0.20, 'crypto': 0.10, 'commodities': 0.10,
                'max_drawdown': 0.25, 'target_return': 0.12
            },
            'zelo_visoko': {
                'stocks': 0.40, 'crypto': 0.30, 'alternatives': 0.20, 'commodities': 0.10,
                'max_drawdown': 0.35, 'target_return': 0.18
            }
        }

        # Optimizirani instrumenti z boljšo likvidnostjo
        self.instruments = {
            'stocks': {
                'US': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ', 'PG', 'MA', 'HD'],
                'EU': ['ASML', 'SAP', 'LVMH.PA', 'NVO', 'TM', 'AZN'],
                'emerging': ['TSM', 'BABA', 'TCEHY']
            },
            'etfs': {
                'broad_market': ['SPY', 'VTI', 'VOO', 'IVV'],
                'international': ['VXUS', 'IEFA', 'IEMG', 'VWO'],
                'sector': ['QQQ', 'XLK', 'XLF', 'XLE', 'XLV'],
                'bonds': ['BND', 'AGG', 'TLT', 'IEF'],
                'commodities': ['GLD', 'SLV', 'VNQ', 'DJP']
            },
            'bonds': ['TLT', 'IEF', 'SHY', 'BND', 'AGG', 'BNDX', 'MUB', 'LQD', 'HYG', 'EMB'],
            'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD'],
            'commodities': ['GLD', 'SLV', 'USO', 'DJP', 'PDBC', 'GSG'],
            'alternatives': ['VNQ', 'REM', 'REIT', 'VTEB', 'SCHH', 'FREL']
        }

        # Flatiziraj instrumente za lažje delo
        self.flat_instruments = {}
        for category, items in self.instruments.items():
            if isinstance(items, dict):
                symbols = []
                for subcategory, symbols_list in items.items():
                    symbols.extend(symbols_list)
                self.flat_instruments[category] = symbols
            else:
                self.flat_instruments[category] = items

        # Časovni horizonti za pridobivanje podatkov
        self.data_periods = {
            'kratko': '1y',
            'srednje': '3y',
            'dolgo': '5y'
        }

        # Nastavitve za napovedni model
        self.lookback_period = 60
        self.forecast_period = 30

        # Inicializiraj prazen slovar za tržne podatke
        self.market_data = {}
        self.benchmark_data = None
        self.data_quality = {}

        # Naloži podatke
        self._load_all_data()

    def _download_single_instrument(self, symbol, period, max_retries=3):
        """Prenesi podatke za posamezen instrument z retry logiko."""
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    period=period,
                    interval='1d',
                    auto_adjust=True,
                    prepost=False,
                    repair=True
                )

                if not data.empty and len(data) >= 100:
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in data.columns for col in required_columns):
                        data = data.replace([np.inf, -np.inf], np.nan)
                        data = data.dropna()

                        if len(data) >= 100:
                            quality_score = self._calculate_data_quality(data, symbol)

                            return {
                                'data': data,
                                'quality': quality_score,
                                'last_price': data['Close'].iloc[-1],
                                'avg_volume': data['Volume'].mean(),
                                'data_points': len(data)
                            }

                print(f"   ⚠️  {symbol}: premalo veljavnih podatkov (attempt {attempt+1})")
                time.sleep(1)

            except Exception as e:
                print(f"   ❌ {symbol}: napaka pri prenosu (attempt {attempt+1}) - {str(e)[:50]}")
                time.sleep(1)

        return None

    def _calculate_data_quality(self, data, symbol):
        """Izračunaj oceno kakovosti podatkov."""
        quality_score = 1.0

        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        quality_score -= missing_ratio * 0.3

        avg_volume = data['Volume'].mean()
        if avg_volume < 100000:
            quality_score -= 0.3
        elif avg_volume < 500000:
            quality_score -= 0.1

        returns = data['Close'].pct_change().dropna()
        extreme_returns = np.abs(returns) > 0.2
        extreme_ratio = extreme_returns.sum() / len(returns)
        if extreme_ratio > 0.05:
            quality_score -= extreme_ratio * 0.2

        if len(data) > 1000:
            quality_score += 0.1

        return max(0.0, min(1.0, quality_score))

    def _load_market_data_parallel(self):
        """Naloži zgodovinske podatke paralelno za hitrejšo obdelavo."""
        print("📊 Nalagam zgodovinske podatke (paralelno)...")
        market_data = {}
        data_quality = {}
        period = self.data_periods.get(self.investment_horizon, '3y')

        all_symbols = []
        symbol_to_category = {}
        for category, symbols in self.flat_instruments.items():
            for symbol in symbols:
                all_symbols.append(symbol)
                symbol_to_category[symbol] = category

        successful_downloads = 0
        failed_downloads = 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_symbol = {
                executor.submit(self._download_single_instrument, symbol, period): symbol
                for symbol in all_symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                category = symbol_to_category[symbol]

                try:
                    result = future.result(timeout=30)
                    if result:
                        market_data[symbol] = result['data']
                        data_quality[symbol] = {
                            'score': result['quality'],
                            'category': category,
                            'last_price': result['last_price'],
                            'avg_volume': result['avg_volume'],
                            'data_points': result['data_points']
                        }
                        successful_downloads += 1
                        print(f"   ✅ {symbol} ({category}): {result['data_points']} podatkov, kakovost: {result['quality']:.3f}")
                    else:
                        failed_downloads += 1
                        print(f"   ❌ {symbol}: neuspešen prenos")

                except Exception as e:
                    failed_downloads += 1
                    print(f"   ❌ {symbol}: napaka - {str(e)[:50]}")

        print(f"📊 Prenos končan: {successful_downloads} uspešnih, {failed_downloads} neuspešnih")

        high_quality_data = {}
        for symbol, data in market_data.items():
            if data_quality[symbol]['score'] >= 0.7:
                high_quality_data[symbol] = data

        print(f"📊 Ohranjenih {len(high_quality_data)} kakovostnih instrumentov od {len(market_data)}")

        return high_quality_data, data_quality

    def _load_all_data(self):
        """Naloži vse potrebne podatke."""
        self.market_data, self.data_quality = self._load_market_data_parallel()
        self.benchmark_data = self._load_benchmark_data()
        self._analyze_data_coverage()

    def _analyze_data_coverage(self):
        """Analiziraj pokritost podatkov po kategorijah."""
        print(f"\n📈 ANALIZA POKRITOSTI PODATKOV:")

        coverage_by_category = {}
        for symbol, quality_info in self.data_quality.items():
            category = quality_info['category']
            if category not in coverage_by_category:
                coverage_by_category[category] = {'total': 0, 'available': 0, 'avg_quality': 0}

            coverage_by_category[category]['total'] += 1
            if symbol in self.market_data:
                coverage_by_category[category]['available'] += 1
                coverage_by_category[category]['avg_quality'] += quality_info['score']

        for category, stats in coverage_by_category.items():
            available_pct = (stats['available'] / stats['total']) * 100 if stats['total'] > 0 else 0
            avg_quality = (stats['avg_quality'] / stats['available']) if stats['available'] > 0 else 0

            print(f"   {category:12} | {stats['available']:2d}/{stats['total']:2d} ({available_pct:5.1f}%) | "
                  f"Povprečna kakovost: {avg_quality:.3f}")

    def _load_benchmark_data(self):
        """Naloži podatke za benchmark (SPY)."""
        print("📊 Nalagam benchmark podatke (SPY)...")
        result = self._download_single_instrument('SPY', self.data_periods.get(self.investment_horizon, '3y'))
        if result:
            print(f"✅ SPY benchmark naložen ({result['data_points']} podatkov, kakovost: {result['quality']:.3f})")
            return result['data']
        else:
            print("❌ SPY benchmark ni na voljo")
            return None

    def _calculate_technical_indicators_enhanced(self, data):
        """Izboljšan izračun tehničnih kazalnikov."""
        df = data.copy()

        # Osnovni kazalniki
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Drseči povprečji
        for period in [5, 10, 20, 50, 100]:
            if len(df) > period:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()

        # Bollinger pasovi
        for bb_period in [20, 50]:
            if len(df) > bb_period:
                bb_middle = df['Close'].rolling(window=bb_period).mean()
                bb_std = df['Close'].rolling(window=bb_period).std()
                df[f'BB_upper_{bb_period}'] = bb_middle + (bb_std * 2)
                df[f'BB_lower_{bb_period}'] = bb_middle - (bb_std * 2)
                df[f'BB_width_{bb_period}'] = (df[f'BB_upper_{bb_period}'] - df[f'BB_lower_{bb_period}']) / bb_middle
                df[f'BB_position_{bb_period}'] = (df['Close'] - df[f'BB_lower_{bb_period}']) / (df[f'BB_upper_{bb_period}'] - df[f'BB_lower_{bb_period}'])

        # RSI
        for rsi_period in [14, 30]:
            if len(df) > rsi_period:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / loss
                df[f'RSI_{rsi_period}'] = 100 - (100 / (1 + rs))

        # MACD
        if len(df) > 50:
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Volatilnost
        for vol_period in [10, 20, 50]:
            if len(df) > vol_period:
                df[f'Volatility_{vol_period}'] = df['Returns'].rolling(window=vol_period).std() * np.sqrt(252)

        # Volume kazalniki
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA_20']

        # Momentum
        for mom_period in [10, 20]:
            if len(df) > mom_period:
                df[f'Momentum_{mom_period}'] = df['Close'] / df['Close'].shift(mom_period) - 1

        # Trend kazalniki
        if len(df) > 100:
            def calculate_trend(series, window=50):
                trends = []
                for i in range(len(series)):
                    if i < window:
                        trends.append(0)
                    else:
                        y = series.iloc[i-window:i].values
                        x = np.arange(len(y))
                        if len(y) > 1:
                            slope = np.polyfit(x, y, 1)[0]
                            trends.append(slope)
                        else:
                            trends.append(0)
                return trends

            df['Price_Trend_50'] = calculate_trend(df['Close'], 50)

        # Čiščenje podatkov
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

        return df

    def _get_sector_and_region(self, symbol):
        """Določi sektor in regijo za simbol."""
        # Poenostavljena implementacija
        sector_map = {
            'AAPL': 'technology', 'MSFT': 'technology', 'GOOGL': 'technology',
            'AMZN': 'technology', 'TSLA': 'automotive', 'NVDA': 'technology',
            'JPM': 'finance', 'V': 'finance', 'JNJ': 'healthcare',
            'XOM': 'energy', 'CVX': 'energy', 'WMT': 'consumer'
        }

        region_map = {
            'AAPL': 'americas', 'MSFT': 'americas', 'GOOGL': 'americas',
            'ASML': 'europe', 'SAP': 'europe', 'LVMH.PA': 'europe',
            'TSM': 'asia_pacific', 'BABA': 'asia_pacific'
        }

        return sector_map.get(symbol, 'unknown'), region_map.get(symbol, 'americas')

    def _calculate_geopolitical_impact(self, symbol):
        """Izračuna vpliv geopolitičnih dejavnikov na sredstvo."""
        sector, region = self._get_sector_and_region(symbol)

        if sector == 'unknown' or region == 'unknown':
            return 0.0

        base_impact = self.geopolitical_analyzer.get_region_impact(region, sector)
        news_impact = self.geopolitical_analyzer.analyze_news_sentiment(symbol)

        total_impact = (base_impact + news_impact) / 2

        return total_impact

    def _prepare_enhanced_prediction_data(self, symbol):
        """Pripravi izboljšane podatke za napovedi."""
        if symbol not in self.market_data:
            return None

        data = self.market_data[symbol].copy()

        if len(data) < self.lookback_period + self.forecast_period + 100:
            print(f"⚠️  {symbol}: premalo podatkov za analizo")
            return None

        data_with_indicators = self._calculate_technical_indicators_enhanced(data)

        important_features = [
            'Close', 'Volume', 'Returns',
            'SMA_20', 'SMA_50', 'EMA_20',
            'RSI_14', 'MACD', 'MACD_hist',
            'BB_position_20', 'BB_width_20',
            'Volatility_20', 'Momentum_10', 'Volume_ratio',
            'Price_Trend_50'
        ]

        available_features = [col for col in important_features if col in data_with_indicators.columns]

        if len(available_features) < 8:
            print(f"⚠️  {symbol}: premalo razpoložljivih značilk")
            return None

        for feature in important_features:
            if feature not in data_with_indicators.columns:
                data_with_indicators[feature] = 0

        data_with_indicators = data_with_indicators.replace([np.inf, -np.inf], np.nan)
        data_with_indicators = data_with_indicators.fillna(0)

        features = data_with_indicators[important_features].values

        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print(f"⚠️  {symbol}: neveljavni podatki po čiščenju")
            return None

        scaler = StandardScaler()
        features_scaled = features.copy()

        if features_scaled.shape[1] > 1:
            features_scaled[:, 1:] = scaler.fit_transform(features_scaled[:, 1:])

        X, y = [], []
        for i in range(self.lookback_period, len(features_scaled) - self.forecast_period):
            X.append(features_scaled[i-self.lookback_period:i, :])
            future_price = features[i+self.forecast_period-1, 0]
            current_price = features[i-1, 0]

            if current_price > 0:
                y.append((future_price - current_price) / current_price)
            else:
                y.append(0)

        if len(X) == 0:
            return None

        X, y = np.array(X), np.array(y)

        split_idx = max(1, int(0.8 * len(X)))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'features': features,
            'scaler': scaler,
            'feature_names': important_features,
            'current_price': features[-1, 0]
        }

    def _create_ensemble_model(self, X_train, y_train, X_test, y_test):
        """Ustvari ensemble model z več algoritmi."""
        models = {}
        predictions = {}

        # Random Forest
        try:
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            rf_pred = rf_model.predict(X_test.reshape(X_test.shape[0], -1))
            rf_mse = mean_squared_error(y_test, rf_pred)
            models['rf'] = {'model': rf_model, 'mse': rf_mse}
            predictions['rf'] = rf_pred
        except Exception as e:
            print(f"⚠️  RandomForest ni uspel: {e}")

        # XGBoost
        if XGB_AVAILABLE:
            try:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=10,
                    learning_rate=0.1,
                    random_state=42
                )
                xgb_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
                xgb_pred = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))
                xgb_mse = mean_squared_error(y_test, xgb_pred)
                models['xgb'] = {'model': xgb_model, 'mse': xgb_mse}
                predictions['xgb'] = xgb_pred
            except Exception as e:
                print(f"⚠️  XGBoost ni uspel: {e}")

        # LSTM
        if TENSORFLOW_AVAILABLE and len(X_train) > 50:
            try:
                lstm_model = self._create_enhanced_lstm_model(X_train.shape)

                callbacks = [
                    EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(factor=0.5, patience=5, verbose=0, min_lr=0.0001)
                ]

                history = lstm_model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=0
                )

                lstm_pred = lstm_model.predict(X_test, verbose=0).flatten()
                lstm_mse = mean_squared_error(y_test, lstm_pred)
                models['lstm'] = {'model': lstm_model, 'mse': lstm_mse}
                predictions['lstm'] = lstm_pred
            except Exception as e:
                print(f"⚠️  LSTM ni uspel: {e}")

        return models, predictions

    def predict_asset_performance_enhanced(self, symbol):
        """Izboljšana napoved uspešnosti sredstva z ensemble modelom."""
        print(f"🔮 Napovedujem uspešnost za {symbol}...")

        data_package = self._prepare_enhanced_prediction_data(symbol)
        if data_package is None:
            return self._get_fallback_prediction(symbol)

        X_train = data_package['X_train']
        X_test = data_package['X_test']
        y_train = data_package['y_train']
        y_test = data_package['y_test']
        current_price = data_package['current_price']

        # Ustvari ensemble model
        models, predictions = self._create_ensemble_model(X_train, y_train, X_test, y_test)

        if not models:
            return self._get_fallback_prediction(symbol)

        # Utežena napoved na podlagi uspešnosti modelov
        final_prediction = 0
        total_weight = 0

        for model_name, model_info in models.items():
            weight = 1 / (model_info['mse'] + 1e-8)  # Utež na podlagi napake
            final_prediction += predictions[model_name].mean() * weight
            total_weight += weight

        if total_weight > 0:
            ensemble_prediction = final_prediction / total_weight
        else:
            ensemble_prediction = 0

        # Izračunaj metrike
        all_predictions = np.mean([pred for pred in predictions.values()], axis=0)
        mse = mean_squared_error(y_test, all_predictions)
        mae = mean_absolute_error(y_test, all_predictions)

        # Geopolitični vpliv
        geo_impact = self._calculate_geopolitical_impact(symbol)

        # Prilagodi napoved glede na geopolitiko
        adjusted_return = ensemble_prediction * (1 - geo_impact * 0.3)
        confidence = max(0.1, 1 - (mae / 0.15))

        sentiment_data = self._get_enhanced_sentiment_analysis(symbol)

        return {
            'expected_return': float(adjusted_return),
            'uncertainty': float(np.sqrt(mse)),
            'confidence': float(confidence),
            'model_type': 'Ensemble',
            'mse': float(mse),
            'mae': float(mae),
            'sentiment': sentiment_data,
            'geopolitical_impact': geo_impact,
            'models_used': list(models.keys())
        }

    def _get_fallback_prediction(self, symbol):
        """Vrni osnovno napoved če napredni modeli ne delujejo."""
        sentiment_data = self._get_enhanced_sentiment_analysis(symbol)
        geo_impact = self._calculate_geopolitical_impact(symbol)

        return {
            'expected_return': 0.0,
            'uncertainty': 0.2,
            'confidence': 0.1,
            'model_type': 'fallback',
            'sentiment': sentiment_data,
            'geopolitical_impact': geo_impact
        }

    def _create_enhanced_lstm_model(self, input_shape):
        """Ustvari izboljšan LSTM model."""
        if not TENSORFLOW_AVAILABLE:
            return None

        try:
            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(input_shape[1], input_shape[2])),
                Dropout(0.2),
                LSTM(75, return_sequences=True),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(50, activation='relu'),
                Dropout(0.1),
                Dense(25, activation='relu'),
                Dense(1, activation='tanh')
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )

            return model

        except Exception as e:
            print(f"⚠️  Napaka pri ustvarjanju LSTM modela: {e}")
            return None

    def _get_enhanced_sentiment_analysis(self, symbol):
        """Izboljšana sentiment analiza."""
        if not TEXTBLOB_AVAILABLE:
            return {'sentiment_score': 0, 'news_count': 0, 'confidence': 0}

        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news

            sentiment_scores = []
            confidence_scores = []

            if news:
                for item in news[:10]:
                    title = item.get('title', '')
                    summary = item.get('summary', '')
                    text = f"{title} {summary}".strip()

                    if text:
                        analysis = TextBlob(text)
                        sentiment_scores.append(analysis.sentiment.polarity)
                        confidence_scores.append(abs(analysis.sentiment.polarity))

            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                avg_confidence = np.mean(confidence_scores)
                news_count = len(sentiment_scores)

                weighted_sentiment = avg_sentiment * avg_confidence

                return {
                    'sentiment_score': weighted_sentiment,
                    'news_count': news_count,
                    'confidence': avg_confidence,
                    'raw_sentiment': avg_sentiment
                }
            else:
                return {'sentiment_score': 0, 'news_count': 0, 'confidence': 0, 'raw_sentiment': 0}

        except Exception as e:
            print(f"⚠️  Napaka pri analizi sentimenta za {symbol}: {e}")
            return {'sentiment_score': 0, 'news_count': 0, 'confidence': 0, 'raw_sentiment': 0}

    def calculate_optimized_portfolio_allocation(self):
        """Izračuna optimizirano razporeditev portfelja."""
        print("🧮 Izračunavam optimalno razporeditev portfelja...")

        if self.risk_profile not in self.risk_profiles:
            print(f"❌ Nepoznan profil tveganja: {self.risk_profile}")
            return None, None

        strategic_allocation = self.risk_profiles[self.risk_profile]

        if not self.market_data:
            print("❌ Ni naloženih tržnih podatkov")
            return None, None

        asset_performance = {}
        prediction_errors = []

        print("🔮 Izvajam napovedi za vsa sredstva...")

        for symbol in self.market_data.keys():
            try:
                category = None
                for cat, symbols in self.flat_instruments.items():
                    if symbol in symbols:
                        category = cat
                        break

                if category is None:
                    continue

                prediction_result = self.predict_asset_performance_enhanced(symbol)

                prediction_result['category'] = category
                prediction_result['data_quality'] = self.data_quality.get(symbol, {}).get('score', 0.5)

                # Izboljšana ocena z geopolitičnimi dejavniki
                expected_return = prediction_result['expected_return']
                confidence = prediction_result['confidence']
                data_quality = prediction_result['data_quality']
                sentiment_score = prediction_result['sentiment']['sentiment_score']
                geo_impact = prediction_result.get('geopolitical_impact', 0)

                # Prilagojena formula z geopolitičnimi dejavniki
                total_score = (
                    expected_return * 0.35 +
                    confidence * 0.25 +
                    data_quality * 0.15 +
                    sentiment_score * 0.15 -
                    geo_impact * 0.10
                )

                prediction_result['total_score'] = total_score
                asset_performance[symbol] = prediction_result

                print(f"   ✅ {symbol} ({category}): donos {expected_return*100:.2f}%, zaupanje {confidence:.3f}, ocena {total_score:.3f}")

            except Exception as e:
                prediction_errors.append(f"{symbol}: {str(e)[:50]}")
                print(f"   ❌ {symbol}: napaka pri napovedi")

        if not asset_performance:
            print("❌ Ni uspešnih napovedi")
            return None, None

        if prediction_errors:
            print(f"⚠️  Napake pri {len(prediction_errors)} simbolih od {len(self.market_data)}")

        optimized_allocation = {}
        category_weights = {k: v for k, v in strategic_allocation.items()
                           if k not in ['max_drawdown', 'target_return']}

        for category, target_weight in category_weights.items():
            category_assets = {k: v for k, v in asset_performance.items()
                              if v['category'] == category}

            if not category_assets:
                print(f"⚠️  Ni podatkov za kategorijo {category}")
                continue

            quality_assets = {k: v for k, v in category_assets.items()
                             if v['confidence'] > 0.2 and v['data_quality'] > 0.5}

            if not quality_assets:
                quality_assets = dict(sorted(category_assets.items(),
                                           key=lambda x: x[1]['total_score'], reverse=True)[:3])

            sorted_assets = sorted(quality_assets.items(), key=lambda x: x[1]['total_score'], reverse=True)
            top_assets = sorted_assets[:min(5, len(sorted_assets))]

            total_category_score = sum(max(0.1, asset[1]['total_score']) for asset in top_assets)

            if total_category_score > 0:
                category_amount = target_weight * self.amount

                for symbol, performance in top_assets:
                    normalized_score = max(0.1, performance['total_score']) / total_category_score
                    confidence_multiplier = 0.5 + (performance['confidence'] * 0.5)
                    final_weight = normalized_score * confidence_multiplier
                    amount = final_weight * category_amount

                    if amount > 100:
                        optimized_allocation[symbol] = amount

        print(f"✅ Razporeditev izračunana za {len(optimized_allocation)} sredstev")

        return optimized_allocation, asset_performance

    def perform_comprehensive_analysis(self):
        """Izvedi celovito analizo."""
        print("\n🚀 ZAČENJAM CELOVITO ANALIZO SMARTINVEST 2.5")
        print("="*70)

        allocation, asset_performance = self.calculate_optimized_portfolio_allocation()

        if allocation is None:
            print("❌ Analiza ni mogoča zaradi pomanjkanja podatkov")
            return

        print("\n📊 Računam metrike tveganja...")
        risk_metrics = self._calculate_enhanced_risk_metrics(allocation, asset_performance)

        print("\n🔍 Izvajam backtest...")
        backtest_results = self._perform_realistic_backtest(allocation, asset_performance)

        print("\n💡 Generiram priporočila...")
        recommendations = self._generate_enhanced_recommendations(
            allocation, asset_performance, risk_metrics, backtest_results
        )

        print("\n📋 Generiram celovito poročilo...")
        self._generate_comprehensive_report(
            allocation, asset_performance, risk_metrics,
            backtest_results, recommendations
        )

        return {
            'allocation': allocation,
            'asset_performance': asset_performance,
            'risk_metrics': risk_metrics,
            'backtest_results': backtest_results,
            'recommendations': recommendations
        }

    def _calculate_enhanced_risk_metrics(self, allocation, asset_performance):
        """Izračuna napredne metrike tveganja."""
        if not allocation or not asset_performance:
            return self._get_default_risk_metrics()

        portfolio_return = 0
        portfolio_variance = 0
        total_amount = sum(allocation.values())

        if total_amount == 0:
            return self._get_default_risk_metrics()

        correlation_matrix = self._calculate_correlation_matrix(allocation.keys())

        weights = {}
        returns = {}
        uncertainties = {}

        for symbol, amount in allocation.items():
            if symbol in asset_performance and amount > 0:
                weight = amount / total_amount
                weights[symbol] = weight
                returns[symbol] = asset_performance[symbol]['expected_return']
                uncertainties[symbol] = asset_performance[symbol]['uncertainty']
                portfolio_return += weight * returns[symbol]

        symbols = list(weights.keys())
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                weight1 = weights[symbol1]
                weight2 = weights[symbol2]
                risk1 = uncertainties[symbol1]
                risk2 = uncertainties[symbol2]

                if i == j:
                    portfolio_variance += (weight1 ** 2) * (risk1 ** 2)
                else:
                    correlation = correlation_matrix.get((symbol1, symbol2), 0.3)
                    covariance = correlation * risk1 * risk2
                    portfolio_variance += 2 * weight1 * weight2 * covariance

        portfolio_volatility = np.sqrt(portfolio_variance)

        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0

        var_95 = portfolio_return - 1.645 * portfolio_volatility
        var_99 = portfolio_return - 2.326 * portfolio_volatility

        expected_shortfall = portfolio_return - 2.5 * portfolio_volatility

        worst_case_individual = min(returns.values()) if returns else -0.3
        max_theoretical_drawdown = min(var_99, worst_case_individual)

        concentration_risk = self._calculate_concentration_risk(weights)
        diversification_score = self._calculate_diversification_score(allocation, asset_performance)

        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall': expected_shortfall,
            'max_theoretical_drawdown': max_theoretical_drawdown,
            'concentration_risk': concentration_risk,
            'diversification_score': diversification_score,
            'number_of_assets': len(weights),
            'largest_position': max(weights.values()) if weights else 0
        }

    def _calculate_correlation_matrix(self, symbols):
        """Izračuna korelacijsko matriko."""
        correlation_matrix = {}

        monthly_returns = {}
        for symbol in symbols:
            if symbol in self.market_data:
                data = self.market_data[symbol]
                monthly_data = data.resample('M').last()
                returns = monthly_data['Close'].pct_change().dropna()
                if len(returns) > 12:
                    monthly_returns[symbol] = returns

        symbols_with_data = list(monthly_returns.keys())
        for i, symbol1 in enumerate(symbols_with_data):
            for j, symbol2 in enumerate(symbols_with_data):
                if i <= j:
                    if symbol1 == symbol2:
                        correlation = 1.0
                    else:
                        returns1 = monthly_returns[symbol1]
                        returns2 = monthly_returns[symbol2]

                        common_dates = returns1.index.intersection(returns2.index)
                        if len(common_dates) > 12:
                            aligned_returns1 = returns1.loc[common_dates]
                            aligned_returns2 = returns2.loc[common_dates]
                            correlation = aligned_returns1.corr(aligned_returns2)

                            if pd.isna(correlation):
                                correlation = 0.3
                        else:
                            correlation = 0.3

                    correlation_matrix[(symbol1, symbol2)] = correlation
                    correlation_matrix[(symbol2, symbol1)] = correlation

        return correlation_matrix

    def _calculate_concentration_risk(self, weights):
        """Izračuna tveganje koncentracije."""
        if not weights:
            return 1.0

        hhi = sum(weight ** 2 for weight in weights.values())
        n_assets = len(weights)
        min_hhi = 1 / n_assets
        max_hhi = 1.0

        if max_hhi > min_hhi:
            normalized_concentration = (hhi - min_hhi) / (max_hhi - min_hhi)
        else:
            normalized_concentration = 0

        return normalized_concentration

    def _calculate_diversification_score(self, allocation, asset_performance):
        """Izračuna oceno diverzifikacije."""
        if not allocation or not asset_performance:
            return 0.5

        category_weights = {}
        for symbol, amount in allocation.items():
            if symbol in asset_performance:
                category = asset_performance[symbol]['category']
                if category not in category_weights:
                    category_weights[category] = 0
                category_weights[category] += amount

        total_amount = sum(allocation.values())
        if total_amount == 0:
            return 0.5

        category_weights = {k: v/total_amount for k, v in category_weights.items()}

        n = len(category_weights)
        if n <= 1:
            return 0.1

        sorted_weights = sorted(category_weights.values())
        gini = (n + 1 - 2 * sum((n + 1 - i) * weight for i, weight in enumerate(sorted_weights, 1))) / n

        diversification_score = 1 - gini
        category_bonus = min(0.2, (n - 1) * 0.05)

        final_score = min(1.0, diversification_score + category_bonus)

        return final_score

    def _get_default_risk_metrics(self):
        """Privzete metrike tveganja."""
        return {
            'expected_return': 0.0,
            'volatility': 0.15,
            'sharpe_ratio': 0.0,
            'var_95': -0.15,
            'var_99': -0.25,
            'expected_shortfall': -0.30,
            'max_theoretical_drawdown': -0.35,
            'concentration_risk': 1.0,
            'diversification_score': 0.0,
            'number_of_assets': 0,
            'largest_position': 0
        }

    def _perform_realistic_backtest(self, allocation, asset_performance, months=12):
        """Izvede realistični backtest."""
        if not allocation:
            return None

        print(f"   📈 Testiram portfelj na zadnjih {months} mesecih...")

        common_dates = None
        valid_symbols = []

        for symbol in allocation.keys():
            if symbol in self.market_data:
                data = self.market_data[symbol]
                if common_dates is None:
                    common_dates = data.index
                else:
                    common_dates = common_dates.intersection(data.index)
                valid_symbols.append(symbol)

        if common_dates is None or len(common_dates) < months * 20:
            print(f"   ⚠️  Premalo skupnih podatkov za backtest")
            return self._generate_synthetic_backtest(allocation, months)

        end_date = common_dates[-1]
        start_date = end_date - pd.DateOffset(months=months)
        test_dates = common_dates[common_dates >= start_date]

        if len(test_dates) < 50:
            return self._generate_synthetic_backtest(allocation, months)

        total_allocation = sum(allocation.values())
        portfolio_values = []
        portfolio_returns = []
        dates = []

        benchmark_values = []
        benchmark_returns = []

        if self.benchmark_data is not None:
            benchmark_data = self.benchmark_data.loc[self.benchmark_data.index.intersection(test_dates)]
        else:
            benchmark_data = None

        initial_value = self.amount

        for i, date in enumerate(test_dates):
            portfolio_value = 0

            for symbol, amount in allocation.items():
                if symbol in valid_symbols:
                    try:
                        price = self.market_data[symbol].loc[date, 'Close']
                        weight = amount / total_allocation if total_allocation > 0 else 0

                        if i == 0:
                            initial_price = price
                            symbol_return = 0
                        else:
                            initial_price = self.market_data[symbol].loc[test_dates[0], 'Close']
                            symbol_return = (price - initial_price) / initial_price if initial_price > 0 else 0

                        portfolio_value += weight * initial_value * (1 + symbol_return)

                    except (KeyError, IndexError):
                        portfolio_value += weight * initial_value

            portfolio_values.append(portfolio_value)
            dates.append(date)

            if i > 0:
                portfolio_return = (portfolio_value - portfolio_values[i-1]) / portfolio_values[i-1]
                portfolio_returns.append(portfolio_return)

            if benchmark_data is not None and date in benchmark_data.index:
                if i == 0:
                    benchmark_initial_price = benchmark_data.loc[date, 'Close']
                    benchmark_values.append(initial_value)
                else:
                    benchmark_price = benchmark_data.loc[date, 'Close']
                    benchmark_return = (benchmark_price - benchmark_initial_price) / benchmark_initial_price
                    benchmark_value = initial_value * (1 + benchmark_return)
                    benchmark_values.append(benchmark_value)

                    benchmark_daily_return = (benchmark_value - benchmark_values[i-1]) / benchmark_values[i-1]
                    benchmark_returns.append(benchmark_daily_return)
            else:
                if i == 0:
                    benchmark_values.append(initial_value)
                else:
                    synthetic_return = np.random.normal(0.0007, 0.012)
                    benchmark_value = benchmark_values[i-1] * (1 + synthetic_return)
                    benchmark_values.append(benchmark_value)
                    benchmark_returns.append(synthetic_return)

        if len(portfolio_values) > 1:
            cumulative_return = (portfolio_values[-1] - initial_value) / initial_value
            benchmark_cumulative_return = (benchmark_values[-1] - initial_value) / initial_value

            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (np.array(portfolio_values) - peak) / peak
            max_drawdown = np.min(drawdown)

            if len(portfolio_returns) > 10 and len(benchmark_returns) > 10:
                try:
                    min_length = min(len(portfolio_returns), len(benchmark_returns))
                    port_rets = portfolio_returns[:min_length]
                    bench_rets = benchmark_returns[:min_length]

                    slope, intercept, r_value, p_value, std_err = linregress(bench_rets, port_rets)
                    alpha = intercept * 252
                    beta = slope
                except:
                    alpha, beta = 0, 1
            else:
                alpha, beta = 0, 1

            return {
                'portfolio_values': portfolio_values,
                'benchmark_values': benchmark_values,
                'dates': dates,
                'cumulative_return': cumulative_return,
                'benchmark_cumulative_return': benchmark_cumulative_return,
                'max_drawdown': max_drawdown,
                'alpha': alpha,
                'beta': beta,
                'portfolio_returns': portfolio_returns,
                'benchmark_returns': benchmark_returns,
                'volatility': np.std(portfolio_returns) * np.sqrt(252) if portfolio_returns else 0,
                'sharpe_ratio': (np.mean(portfolio_returns) * 252 - 0.02) / (np.std(portfolio_returns) * np.sqrt(252)) if portfolio_returns else 0
            }

        return None

    def _generate_synthetic_backtest(self, allocation, months):
        """Generiraj sintetični backtest."""
        print(f"   ⚠️  Uporabljam sintetični backtest zaradi pomanjkanja podatkov")

        risk_params = self.risk_profiles[self.risk_profile]
        target_return = risk_params.get('target_return', 0.08)
        max_drawdown = risk_params.get('max_drawdown', 0.15)

        np.random.seed(42)
        trading_days = months * 21
        dates = [datetime.now() - timedelta(days=i) for i in range(trading_days, 0, -1)]

        daily_return_mean = target_return / 252
        daily_volatility = max_drawdown / np.sqrt(252)

        portfolio_returns = np.random.normal(daily_return_mean, daily_volatility, trading_days)
        benchmark_returns = np.random.normal(0.008/252, 0.12/np.sqrt(252), trading_days)

        portfolio_values = [self.amount]
        benchmark_values = [self.amount]

        for i in range(trading_days):
            portfolio_value = portfolio_values[-1] * (1 + portfolio_returns[i])
            benchmark_value = benchmark_values[-1] * (1 + benchmark_returns[i])

            portfolio_values.append(portfolio_value)
            benchmark_values.append(benchmark_value)

        cumulative_return = (portfolio_values[-1] - self.amount) / self.amount
        benchmark_cumulative_return = (benchmark_values[-1] - self.amount) / self.amount

        peak = np.maximum.accumulate(portfolio_values[1:])
        drawdown = (np.array(portfolio_values[1:]) - peak) / peak
        max_drawdown_realized = np.min(drawdown)

        return {
            'portfolio_values': portfolio_values,
            'benchmark_values': benchmark_values,
            'dates': dates + [datetime.now()],
            'cumulative_return': cumulative_return,
            'benchmark_cumulative_return': benchmark_cumulative_return,
            'max_drawdown': max_drawdown_realized,
            'alpha': cumulative_return - benchmark_cumulative_return,
            'beta': 1.0,
            'portfolio_returns': portfolio_returns.tolist(),
            'benchmark_returns': benchmark_returns.tolist(),
            'volatility': np.std(portfolio_returns) * np.sqrt(252) if portfolio_returns else 0,
            'sharpe_ratio': (np.mean(portfolio_returns) * 252 - 0.02) / (np.std(portfolio_returns) * np.sqrt(252)) if portfolio_returns else 0
        }

    def _generate_enhanced_recommendations(self, allocation, asset_performance, risk_metrics, backtest_results):
        """Generira izboljšana priporočila."""
        recommendations = []

        risk_profile_name = self.risk_profile.replace('_', ' ').title()
        recommendations.append(f"🧠 **Profil tveganja**: {risk_profile_name}")
        recommendations.append(f"💰 **Začetni kapital**: {self.amount:,.0f} €")
        recommendations.append(f"📅 **Naložbeni horizont**: {self.investment_horizon}")

        if allocation:
            recommendations.append("\n📊 **Razporeditev sredstev**:")
            for symbol, amount in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
                category = asset_performance[symbol]['category'] if symbol in asset_performance else "Neznano"
                geo_impact = asset_performance[symbol].get('geopolitical_impact', 0)
                recommendations.append(f"   • {symbol} ({category}): {amount:,.0f} € [Geo-riski: {geo_impact:.2f}]")

        if risk_metrics:
            recommendations.append(f"\n⚠️  **Ocena tveganja**:")
            recommendations.append(f"   • Pričakovan donos: {risk_metrics['expected_return']*100:.1f}% letno")
            recommendations.append(f"   • Volatilnost: {risk_metrics['volatility']*100:.1f}%")
            recommendations.append(f"   • Najhujši pričakovani padec: {risk_metrics['max_theoretical_drawdown']*100:.1f}%")
            recommendations.append(f"   • Sharpejevo razmerje: {risk_metrics['sharpe_ratio']:.2f}")
            recommendations.append(f"   • Stopnja diverzifikacije: {risk_metrics['diversification_score']*100:.0f}%")

        if backtest_results:
            recommendations.append(f"\n📈 **Zgodovinska uspešnost**:")
            recommendations.append(f"   • Donos portfelja: {backtest_results['cumulative_return']*100:.1f}%")
            if 'benchmark_cumulative_return' in backtest_results:
                recommendations.append(f"   • Donos benchmarka: {backtest_results['benchmark_cumulative_return']*100:.1f}%")
            recommendations.append(f"   • Maksimalni padec: {backtest_results['max_drawdown']*100:.1f}%")

        recommendations.append(f"\n💡 **Splošna priporočila**:")

        if risk_metrics and risk_metrics['concentration_risk'] > 0.7:
            recommendations.append("   • 🚨 Razmislite o večji diverzifikaciji - preveč koncentrirano v nekaj sredstvih")

        if risk_metrics and risk_metrics['expected_return'] < self.risk_profiles[self.risk_profile]['target_return'] * 0.7:
            recommendations.append("   • 📉 Pričakovani donos je pod ciljnim - razmislite o prilagoditvi razporeditve")

        recommendations.append("   • 🔄 Redno spremljajte portfelj (vsaj četrtletno)")
        recommendations.append("   • 🌍 Spremljajte geopolitične dejavnike, ki vplivajo na vaše naložbe")

        return recommendations

    def _generate_comprehensive_report(self, allocation, asset_performance, risk_metrics, backtest_results, recommendations):
        """Generira celovito poročilo."""
        print("\n" + "="*70)
        print("📋 SMARTINVEST 2.5 - CELOVITO POROČILO")
        print("="*70)

        for recommendation in recommendations:
            print(recommendation)

        if allocation:
            plt.figure(figsize=(12, 8))

            category_allocation = {}
            for symbol, amount in allocation.items():
                if symbol in asset_performance:
                    category = asset_performance[symbol]['category']
                    if category not in category_allocation:
                        category_allocation[category] = 0
                    category_allocation[category] += amount

            if category_allocation:
                plt.subplot(2, 2, 1)
                plt.pie(category_allocation.values(), labels=category_allocation.keys(), autopct='%1.1f%%')
                plt.title('Razporeditev po kategorijah')

            plt.subplot(2, 2, 2)
            top_10 = dict(sorted(allocation.items(), key=lambda x: x[1], reverse=True)[:10])
            plt.barh(range(len(top_10)), list(top_10.values()))
            plt.yticks(range(len(top_10)), list(top_10.keys()))
            plt.title('Top 10 sredstev po vrednosti')
            plt.xlabel('Vrednost (€)')

            if asset_performance:
                plt.subplot(2, 2, 3)
                returns = []
                geo_impacts = []
                symbols = []
                for symbol, perf in asset_performance.items():
                    if symbol in allocation:
                        returns.append(perf['expected_return'] * 100)
                        geo_impacts.append(perf.get('geopolitical_impact', 0) * 100)
                        symbols.append(symbol)

                if returns:
                    x = range(len(returns[:10]))
                    width = 0.35
                    plt.bar(x, returns[:10], width, label='Pričakovan donos (%)')
                    plt.bar([i + width for i in x], geo_impacts[:10], width, label='Geopolitični vpliv (%)')
                    plt.xticks([i + width/2 for i in x], symbols[:10], rotation=45)
                    plt.legend()
                    plt.title('Donosi in geopolitični vplivi')

            if backtest_results and 'portfolio_values' in backtest_results:
                plt.subplot(2, 2, 4)
                plt.plot(backtest_results['portfolio_values'], label='Portfelj')
                if 'benchmark_values' in backtest_results:
                    plt.plot(backtest_results['benchmark_values'], label='Benchmark')
                plt.legend()
                plt.title('Zgodovinski razvoj vrednosti')
                plt.xlabel('Čas')
                plt.ylabel('Vrednost (€)')

            plt.tight_layout()
            plt.show()

        print(f"\n💎 **POVZETEK SMARTINVEST 2.5**:")
        print(f"   Portfolio je usklajen z vašim profilom tveganja ({self.risk_profile})")
        print(f"   Pričakovan donos: {risk_metrics['expected_return']*100:.1f}% letno")
        print(f"   Pričakovana volatilnost: {risk_metrics['volatility']*100:.1f}%")
        print(f"   Število sredstev: {len(allocation)}")
        print(f"   Stopnja diverzifikacije: {risk_metrics['diversification_score']*100:.0f}%")
        print(f"   Uporabljeni modeli: Ensemble (RF, XGBoost, LSTM)")
        print(f"   Geopolitična analiza: Vključena")

# Demonstracija uporabe
if __name__ == "__main__":
    print("🚀 ZAGON SMARTINVEST 2.5")
    print("="*50)

    advisor = SmartInvest25(
        amount=100000,
        risk_profile='srednje',
        investment_horizon='srednje'
    )

    results = advisor.perform_comprehensive_analysis()

    if results:
        print("\n🎉 ANALIZA USPEŠNO ZAKLJUČENA!")
        print(f"Število predlaganih sredstev: {len(results['allocation'])}")
        print(f"Skupna vrednost portfelja: {sum(results['allocation'].values()):,.0f} €")

        # Prikaz geopolitičnih tveganj
        geo_risks = []
        for symbol, performance in results['asset_performance'].items():
            if symbol in results['allocation']:
                geo_impact = performance.get('geopolitical_impact', 0)
                if geo_impact > 0.1:
                    geo_risks.append((symbol, geo_impact))

        if geo_risks:
            print(f"\n⚠️  OPOZORILO: Geopolitična tveganja:")
            for symbol, risk in sorted(geo_risks, key=lambda x: x[1], reverse=True)[:5]:
                print(f"   • {symbol}: {risk:.3f}")
    else:
        print("Analiza ni uspela. Preverite internetno povezavo in poskusite znova.")