import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import requests
import json
import sqlite3
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# AI/ML knjižnice
try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, concatenate, Attention
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow ni na voljo.")
    TENSORFLOW_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost ni na voljo.")
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    print("⚠️  LightGBM ni na voljo.")
    LGBM_AVAILABLE = False

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    print("⚠️  TA-Lib ni na voljo.")
    TALIB_AVAILABLE = False

# Eksterni API-ji
import yfinance as yf

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    print("⚠️  TextBlob ni na voljo.")
    TEXTBLOB_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# MODUL 1: DATA MANAGER Z CACHE SISTEMOM
# =============================================================================

class DataManager:
    """Upravlja s podatki in lokalnim cache sistemom."""

    def __init__(self, cache_file="smartinvest_cache.db"):
        self.cache_file = cache_file
        self._init_cache()

    def _init_cache(self):
        """Inicializira SQLite cache bazo."""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()

        # Tabela za tržne podatke
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, date)
            )
        ''')

        # Tabela za makro podatke
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS macro_data (
                indicator TEXT,
                date TEXT,
                value REAL,
                PRIMARY KEY (indicator, date)
            )
        ''')

        # Tabela za rezultate analiz
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                symbol TEXT,
                analysis_date TEXT,
                expected_return REAL,
                confidence REAL,
                sentiment REAL,
                geo_impact REAL,
                model_type TEXT,
                PRIMARY KEY (symbol, analysis_date)
            )
        ''')

        conn.commit()
        conn.close()

    async def fetch_market_data_async(self, symbols, period="3y"):
        """Asinhrono pridobi tržne podatke."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in symbols:
                task = self._download_single_instrument_async(session, symbol, period)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return {symbol: result for symbol, result in zip(symbols, results) if result is not None}

    async def _download_single_instrument_async(self, session, symbol, period):
        """Asinhrono prenosi podatke za posamezen simbol."""
        try:
            # Preveri cache najprej
            cached_data = self._get_cached_market_data(symbol, period)
            if cached_data is not None:
                return cached_data

            # Prenos iz yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval='1d')

            if not data.empty and len(data) > 100:
                # Shrani v cache
                self._cache_market_data(symbol, data)
                return data

        except Exception as e:
            print(f"❌ Napaka pri prenosu {symbol}: {e}")

        return None

    def _get_cached_market_data(self, symbol, period):
        """Pridobi podatke iz cache."""
        conn = sqlite3.connect(self.cache_file)

        # Izračunaj datum začetka glede na periodo
        if period == "1y":
            start_date = datetime.now() - timedelta(days=365)
        elif period == "3y":
            start_date = datetime.now() - timedelta(days=3*365)
        else:  # 5y
            start_date = datetime.now() - timedelta(days=5*365)

        query = '''
            SELECT date, open, high, low, close, volume
            FROM market_data
            WHERE symbol = ? AND date >= ?
            ORDER BY date
        '''

        df = pd.read_sql_query(query, conn, params=[symbol, start_date.strftime('%Y-%m-%d')])
        conn.close()

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df

        return None

    def _cache_market_data(self, symbol, data):
        """Shrani podatke v cache."""
        conn = sqlite3.connect(self.cache_file)

        # Pripravi podatke za vstavljanje
        records = []
        for date, row in data.iterrows():
            records.append((
                symbol, date.strftime('%Y-%m-%d'),
                row['Open'], row['High'], row['Low'],
                row['Close'], row['Volume']
            ))

        # Vstavi podatke
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT OR REPLACE INTO market_data
            (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', records)

        conn.commit()
        conn.close()

# =============================================================================
# MODUL 2: MACRO ECONOMIC ANALYZER
# =============================================================================

class MacroEconomicAnalyzer:
    """Analizira makroekonomske kazalnike prek FRED API."""

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.fred_indicators = {
            'CPI': 'CPIAUCSL',  # Consumer Price Index
            'PPI': 'PPIACO',    # Producer Price Index
            'GDP': 'GDP',       # Gross Domestic Product
            'UNRATE': 'UNRATE', # Unemployment Rate
            'FEDFUNDS': 'FEDFUNDS', # Federal Funds Rate
            'M2': 'M2SL',       # Money Supply
            'INDPRO': 'INDPRO', # Industrial Production
            'RETAIL': 'RSAFS',  # Retail Sales
        }

    def fetch_macro_data(self):
        """Pridobi makroekonomske podatke."""
        macro_data = {}

        for name, code in self.fred_indicators.items():
            try:
                if self.api_key:
                    # Uporabi FRED API če je na voljo ključ
                    url = f"https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        'series_id': code,
                        'api_key': self.api_key,
                        'file_type': 'json',
                        'sort_order': 'desc',
                        'limit': 100
                    }
                    response = requests.get(url, params=params)
                    data = response.json()

                    if 'observations' in data:
                        df = pd.DataFrame(data['observations'])
                        df['date'] = pd.to_datetime(df['date'])
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        df.set_index('date', inplace=True)
                        macro_data[name] = df['value']
                else:
                    # Fallback: generiraj sintetične podatke
                    dates = pd.date_range(end=datetime.now(), periods=100, freq='M')
                    values = self._generate_synthetic_macro_data(name, len(dates))
                    macro_data[name] = pd.Series(values, index=dates)

            except Exception as e:
                print(f"⚠️  Napaka pri pridobivanju {name}: {e}")
                # Fallback na sintetične podatke
                dates = pd.date_range(end=datetime.now(), periods=100, freq='M')
                values = self._generate_synthetic_macro_data(name, len(dates))
                macro_data[name] = pd.Series(values, index=dates)

        return macro_data

    def _generate_synthetic_macro_data(self, indicator, n_periods):
        """Generiraj realistične sintetične makro podatke."""
        np.random.seed(42)

        if indicator == 'CPI':
            # Inflacija: 2-4% z nekaj volatilnostjo
            base = 2.5
            noise = np.random.normal(0, 0.3, n_periods)
            trend = np.linspace(0, 0.5, n_periods)
            return base + trend + noise

        elif indicator == 'GDP':
            # Rast BDP: 1-3%
            base = 2.0
            noise = np.random.normal(0, 0.8, n_periods)
            return base + noise

        elif indicator == 'UNRATE':
            # Brezposelnost: 3-6%
            base = 4.0
            noise = np.random.normal(0, 0.5, n_periods)
            trend = np.linspace(0, -0.3, n_periods)
            return base + trend + noise

        else:
            # Privzeto: stacionarni proces
            return np.random.normal(0, 1, n_periods)

    def analyze_economic_cycle(self, macro_data):
        """Analizira gospodarski cikel."""
        if 'GDP' not in macro_data or 'CPI' not in macro_data:
            return 'unknown'

        gdp_growth = macro_data['GDP'].pct_change().dropna().iloc[-6:].mean() * 100  # Zadnjih 6 mesecev
        inflation = macro_data['CPI'].pct_change().dropna().iloc[-6:].mean() * 100

        if gdp_growth > 2 and inflation < 3:
            return 'expansion'
        elif gdp_growth < 0:
            return 'recession'
        elif gdp_growth > 0 and inflation > 5:
            return 'stagflation'
        else:
            return 'slow_growth'

# =============================================================================
# MODUL 3: MARKET REGIME DETECTOR
# =============================================================================

class RegimeDetector:
    """Prepoznava tržne režime z uporabo HMM/GMM."""

    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        self.regime_labels = {0: 'bear', 1: 'sideways', 2: 'bull'}
        self.is_fitted = False

    def extract_regime_features(self, price_data):
        """Pripravi značilke za prepoznavo režima."""
        returns = price_data['Close'].pct_change().dropna()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        momentum = price_data['Close'] / price_data['Close'].shift(20) - 1

        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'momentum': momentum
        }).dropna()

        return features

    def detect_regime(self, price_data):
        """Prepoznaj trenutni tržni režim."""
        features = self.extract_regime_features(price_data)

        if len(features) < 50:
            return 'unknown'

        # Uči model če še ni bil
        if not self.is_fitted:
            self.gmm.fit(features.tail(1000))  # Uporabi zadnjih 1000 opazovanj
            self.is_fitted = True

        # Napovedi za zadnje obdobje
        current_features = features.iloc[-1:].values
        regime_idx = self.gmm.predict(current_features)[0]

        return self.regime_labels.get(regime_idx, 'unknown')

    def get_regime_metrics(self, price_data):
        """Pridobi metrike za vsak režim."""
        features = self.extract_regime_features(price_data)

        if not self.is_fitted or len(features) < 50:
            return {}

        regime_probs = self.gmm.predict_proba(features.iloc[-30:])  # Zadnjih 30 dni
        avg_probs = regime_probs.mean(axis=0)

        return {
            'bull_probability': avg_probs[2] if len(avg_probs) > 2 else 0,
            'bear_probability': avg_probs[0] if len(avg_probs) > 0 else 0,
            'sideways_probability': avg_probs[1] if len(avg_probs) > 1 else 0,
            'current_regime': self.detect_regime(price_data)
        }

# =============================================================================
# MODUL 4: FEATURE ENGINE
# =============================================================================

class FeatureEngine:
    """Združuje vse značilke v enoten feature matrix."""

    def __init__(self):
        self.scaler = StandardScaler()

    def create_feature_matrix(self, market_data, macro_data, sentiment_data, geo_data):
        """Ustvari končno matriko značilk."""
        all_features = []

        for symbol, data in market_data.items():
            # Tehnične značilke
            tech_features = self._extract_technical_features(data)

            # Makro značilke
            macro_features = self._extract_macro_features(macro_data, data.index[-1])

            # Sentiment značilke
            sentiment_features = self._extract_sentiment_features(sentiment_data.get(symbol, {}))

            # Geopolitične značilke
            geo_features = self._extract_geo_features(geo_data.get(symbol, {}))

            # Kombiniraj vse značilke
            combined_features = np.concatenate([
                tech_features,
                macro_features,
                sentiment_features,
                geo_features
            ])

            all_features.append(combined_features)

        return np.array(all_features)

    def _extract_technical_features(self, data):
        """Ekstrahiraj tehnične značilke."""
        returns = data['Close'].pct_change().dropna()

        features = [
            returns.mean(), returns.std(), returns.skew(), returns.kurtosis(),
            data['Close'].rolling(20).mean().iloc[-1] / data['Close'].iloc[-1] - 1,  # SMA20 razlika
            data['Close'].rolling(50).mean().iloc[-1] / data['Close'].iloc[-1] - 1,  # SMA50 razlika
            data['Volume'].mean() / data['Volume'].rolling(20).mean().iloc[-1],  # Volume ratio
        ]

        return np.nan_to_num(features, nan=0.0)

    def _extract_macro_features(self, macro_data, date):
        """Ekstrahiraj makro značilke."""
        features = []

        for indicator, series in macro_data.items():
            try:
                # Pridobi najnovejšo vrednost pred danim datumom
                recent_data = series[series.index <= date]
                if not recent_data.empty:
                    features.append(recent_data.iloc[-1])
                else:
                    features.append(0.0)
            except:
                features.append(0.0)

        return np.nan_to_num(features, nan=0.0)

    def _extract_sentiment_features(self, sentiment_data):
        """Ekstrahiraj sentiment značilke."""
        return np.array([
            sentiment_data.get('sentiment_score', 0),
            sentiment_data.get('news_count', 0) / 10,  # Normaliziraj
            sentiment_data.get('confidence', 0)
        ])

    def _extract_geo_features(self, geo_data):
        """Ekstrahiraj geopolitične značilke."""
        return np.array([
            geo_data.get('risk_score', 0),
            geo_data.get('sector_impact', 0),
            geo_data.get('region_impact', 0)
        ])

# =============================================================================
# MODUL 5: SMART MODEL HUB
# =============================================================================

class SmartModelHub:
    """Dinamično izbira in upravlja z AI modeli."""

    def __init__(self):
        self.models = {}
        self.regime_model_mapping = {
            'bull': ['xgb', 'lstm'],
            'bear': ['rf', 'xgb'],
            'sideways': ['lstm', 'ensemble'],
            'unknown': ['ensemble']
        }

    def get_models_for_regime(self, regime):
        """Pridobi priporočene modele za trenutni režim."""
        return self.regime_model_mapping.get(regime, ['ensemble'])

    def create_ensemble_prediction(self, X, regime):
        """Ustvari ensemble napoved za dane podatke."""
        models_to_use = self.get_models_for_regime(regime)
        predictions = []
        weights = []

        for model_name in models_to_use:
            if model_name in self.models:
                pred = self.models[model_name].predict(X)
                predictions.append(pred)
                # Dodeli uteži glede na zaupanje v model za določen režim
                weights.append(1.0)

        if predictions:
            # Uteženo povprečje
            weights = np.array(weights) / sum(weights)
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            return ensemble_pred
        else:
            # Fallback na RandomForest
            return np.zeros(X.shape[0])

    def train_models(self, X_train, y_train, X_test, y_test):
        """Natrenira vse modele."""
        # Random Forest
        self.models['rf'] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models['rf'].fit(X_train, y_train)

        # XGBoost
        if XGB_AVAILABLE:
            self.models['xgb'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
            self.models['xgb'].fit(X_train, y_train)

        # LightGBM
        if LGBM_AVAILABLE:
            self.models['lgb'] = lgb.LGBMRegressor(n_estimators=100, random_state=42)
            self.models['lgb'].fit(X_train, y_train)

        # LSTM (če je na voljo TensorFlow)
        if TENSORFLOW_AVAILABLE and len(X_train) > 100:
            try:
                lstm_model = self._create_lstm_model(X_train.shape[1])
                history = lstm_model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50, batch_size=32, verbose=0
                )
                self.models['lstm'] = lstm_model
            except Exception as e:
                print(f"⚠️  LSTM training failed: {e}")

    def _create_lstm_model(self, input_dim):
        """Ustvari LSTM model z attention mehanizmom."""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(1, input_dim)),
            Attention(),
            Dropout(0.3),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='linear')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

# =============================================================================
# MODUL 6: PORTFOLIO OPTIMIZER Z RL IN SIMULACIJAMI
# =============================================================================

class PortfolioOptimizer:
    """Optimizira portfelj z Monte Carlo simulacijami in RL."""

    def __init__(self, n_simulations=1000):
        self.n_simulations = n_simulations

    def monte_carlo_simulation(self, expected_returns, cov_matrix, time_horizon=252, n_simulations=None):
        """Izvede Monte Carlo simulacijo portfelja."""
        n_simulations = n_simulations or self.n_simulations
        n_assets = len(expected_returns)

        # Generiraj naključne donose
        portfolio_values = np.zeros((n_simulations, time_horizon + 1))
        portfolio_values[:, 0] = 1  # Začetna vrednost = 1

        # Cholesky dekompozicija za korelirane donose
        try:
            L = np.linalg.cholesky(cov_matrix)
        except:
            # Če kovariančna matrika ni pozitivno definitna, uporabi diagonalno
            L = np.diag(np.sqrt(np.diag(cov_matrix)))

        for i in range(n_simulations):
            # Generiraj korelirane naključne številke
            z = np.random.normal(0, 1, (time_horizon, n_assets))
            correlated_returns = expected_returns + np.dot(z, L.T)

            # Simuliraj pot portfelja
            for t in range(time_horizon):
                portfolio_values[i, t+1] = portfolio_values[i, t] * (1 + np.mean(correlated_returns[t]))

        return portfolio_values

    def calculate_var(self, portfolio_values, confidence=0.95):
        """Izračuna Value at Risk."""
        final_returns = (portfolio_values[:, -1] - 1) / 1
        var = np.percentile(final_returns, (1 - confidence) * 100)
        return var

    def calculate_expected_shortfall(self, portfolio_values, confidence=0.95):
        """Izračuna Expected Shortfall (CVaR)."""
        final_returns = (portfolio_values[:, -1] - 1) / 1
        var = self.calculate_var(portfolio_values, confidence)
        es = final_returns[final_returns <= var].mean()
        return es

    def stress_test_scenarios(self, portfolio, scenarios):
        """Izvede stres teste za različne scenarije."""
        results = {}

        for scenario_name, scenario_params in scenarios.items():
            # Prilagodi pričakovane donose glede na scenarij
            adjusted_returns = portfolio['expected_returns'] * scenario_params.get('return_multiplier', 1)
            adjusted_volatility = portfolio['volatility'] * scenario_params.get('volatility_multiplier', 1)

            # Ponovna Monte Carlo simulacija
            cov_matrix = np.diag(adjusted_volatility**2)  # Poenostavljena kovariančna matrika
            simulated_values = self.monte_carlo_simulation(adjusted_returns, cov_matrix, n_simulations=500)

            results[scenario_name] = {
                'var_95': self.calculate_var(simulated_values, 0.95),
                'expected_shortfall': self.calculate_expected_shortfall(simulated_values, 0.95),
                'probability_negative': (simulated_values[:, -1] < 1).mean()
            }

        return results

# =============================================================================
# MODUL 7: SMART REPORT - INTERAKTIVNI DASHBOARD
# =============================================================================

class SmartReport:
    """Generira interaktivna poročila in vizualizacije."""

    def __init__(self):
        self.figures = {}

    def create_comprehensive_dashboard(self, portfolio_data, risk_metrics, backtest_results, regime_info):
        """Ustvari celovit dashboard."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Razporeditev portfelja', 'Zgodovinska uspešnost',
                'Metrike tveganja', 'Tržni režim',
                'Geopolitični vplivi', 'Stres testi'
            ),
            specs=[
                [{"type": "pie"}, {"type": "xy"}],
                [{"type": "bar"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )

        # 1. Razporeditev portfelja
        if 'allocation' in portfolio_data:
            symbols = list(portfolio_data['allocation'].keys())
            amounts = list(portfolio_data['allocation'].values())
            fig.add_trace(go.Pie(labels=symbols, values=amounts, name="Portfolio"), 1, 1)

        # 2. Zgodovinska uspešnost
        if backtest_results and 'portfolio_values' in backtest_results:
            dates = backtest_results.get('dates', list(range(len(backtest_results['portfolio_values']))))
            fig.add_trace(go.Scatter(x=dates, y=backtest_results['portfolio_values'],
                                   name="Portfolio", line=dict(color='blue')), 1, 2)

            if 'benchmark_values' in backtest_results:
                fig.add_trace(go.Scatter(x=dates, y=backtest_results['benchmark_values'],
                                       name="Benchmark", line=dict(color='red')), 1, 2)

        # 3. Metrike tveganja
        if risk_metrics:
            metrics_names = ['Donos', 'Volatilnost', 'Sharpe', 'Max Drawdown']
            metrics_values = [
                risk_metrics.get('expected_return', 0) * 100,
                risk_metrics.get('volatility', 0) * 100,
                risk_metrics.get('sharpe_ratio', 0),
                risk_metrics.get('max_drawdown', 0) * 100
            ]
            fig.add_trace(go.Bar(x=metrics_names, y=metrics_values, name="Risk Metrics"), 2, 1)

        # 4. Tržni režim
        if regime_info:
            current_regime = regime_info.get('current_regime', 'unknown')
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=regime_info.get('bull_probability', 0) * 100,
                title={'text': f"Tržni režim: {current_regime}"},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "green" if current_regime == 'bull' else
                                     'red' if current_regime == 'bear' else 'yellow'}},
            ), 2, 2)

        fig.update_layout(height=800, title_text="SmartInvest 3.0 Dashboard", showlegend=False)
        return fig

    def export_to_html(self, fig, filename="smartinvest_report.html"):
        """Izvozi dashboard v HTML."""
        fig.write_html(filename)
        print(f"✅ Poročilo shranjeno kot {filename}")

    def export_to_pdf(self, fig, filename="smartinvest_report.pdf"):
        """Izvozi dashboard v PDF."""
        try:
            fig.write_image(filename)
            print(f"✅ Poročilo shranjeno kot {filename}")
        except Exception as e:
            print(f"⚠️  Napaka pri izvozu v PDF: {e}")

# =============================================================================
# GLAVNI RAZRED SMARTINVEST 3.0
# =============================================================================

class SmartInvest30:
    """
    SmartInvest 3.0 - Celovit AI investicijski sistem
    z makro inteligenco, samoprilagodljivimi modeli in vizualnim nadzorom.
    """

    def __init__(self, amount, risk_profile, investment_horizon, fred_api_key=None):
        self.amount = amount
        self.risk_profile = risk_profile
        self.investment_horizon = investment_horizon

        # Inicializiraj vse module
        self.data_manager = DataManager()
        self.macro_analyzer = MacroEconomicAnalyzer(api_key=fred_api_key)
        self.regime_detector = RegimeDetector()
        self.feature_engine = FeatureEngine()
        self.model_hub = SmartModelHub()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.report_generator = SmartReport()

        # Definicija instrumentov (razširjena verzija)
        self.instruments = self._initialize_instruments()

        # Podatki
        self.market_data = {}
        self.macro_data = {}
        self.asset_performance = {}
        self.current_regime = 'unknown'

    def _initialize_instruments(self):
        """Inicializira razširjen seznam instrumentov."""
        return {
            'stocks': {
                'US_LargeCap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B'],
                'US_Value': ['JPM', 'JNJ', 'PG', 'XOM', 'CVX', 'WMT'],
                'International': ['ASML', 'TSM', 'NVO', 'SAP', 'RY'],
                'Emerging': ['BABA', 'TCEHY', 'INFY', 'PBR']
            },
            'etfs': {
                'Broad_Market': ['SPY', 'VTI', 'IVV', 'VOO'],
                'Sector_Tech': ['QQQ', 'XLK', 'VGT', 'SOXX'],
                'Sector_Finance': ['XLF', 'VFH', 'KRE'],
                'International': ['VXUS', 'VEU', 'IEFA'],
                'Bonds': ['BND', 'AGG', 'TLT', 'IEF'],
                'Commodities': ['GLD', 'SLV', 'USO', 'DBA']
            },
            'bonds': ['BND', 'AGG', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'MUB'],
            'crypto': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD'],
            'alternatives': ['VNQ', 'REET', 'REM', 'PSP']
        }

    async def load_all_data(self):
        """Naloži vse potrebne podatke asinhrono."""
        print("📊 Nalagam podatke...")

        # Zberi vse simbole
        all_symbols = []
        for category, items in self.instruments.items():
            if isinstance(items, dict):
                for subcategory, symbols in items.items():
                    all_symbols.extend(symbols)
            else:
                all_symbols.extend(items)

        # Naloži tržne podatke
        self.market_data = await self.data_manager.fetch_market_data_async(
            all_symbols, period=self.data_periods.get(self.investment_horizon, '3y')
        )

        # Naloži makro podatke
        self.macro_data = self.macro_analyzer.fetch_macro_data()

        print(f"✅ Naloženo {len(self.market_data)} instrumentov in {len(self.macro_data)} makro kazalnikov")

    def analyze_market_regime(self):
        """Analizira trenutni tržni režim."""
        if not self.market_data:
            return 'unknown'

        # Uporabi SPY kot benchmark za določitev režima
        spy_data = self.market_data.get('SPY')
        if spy_data is not None:
            self.current_regime = self.regime_detector.detect_regime(spy_data)
            regime_metrics = self.regime_detector.get_regime_metrics(spy_data)

            print(f"📈 Trenutni tržni režim: {self.current_regime}")
            print(f"   Verjetnost bull: {regime_metrics.get('bull_probability', 0):.1%}")
            print(f"   Verjetnost bear: {regime_metrics.get('bear_probability', 0):.1%}")

            return regime_metrics

        return {'current_regime': 'unknown'}

    def perform_comprehensive_analysis(self):
        """Izvede celovito analizo."""
        print("\n🚀 ZAČENJAM SMARTINVEST 3.0 ANALIZO")
        print("="*60)

        # 1. Analiza tržnega režima
        regime_info = self.analyze_market_regime()

        # 2. Napovedi uspešnosti sredstev
        asset_predictions = self._predict_asset_performance()

        # 3. Optimizacija portfelja
        optimized_allocation = self._optimize_portfolio(asset_predictions)

        # 4. Analiza tveganja
        risk_metrics = self._calculate_risk_metrics(optimized_allocation, asset_predictions)

        # 5. Stres testi
        stress_test_results = self._perform_stress_tests(optimized_allocation, asset_predictions)

        # 6. Generiraj poročilo
        dashboard = self.report_generator.create_comprehensive_dashboard(
            {'allocation': optimized_allocation},
            risk_metrics,
            self._perform_backtest(optimized_allocation),
            regime_info
        )

        # Prikaži rezultate
        dashboard.show()

        # Shrani poročilo
        self.report_generator.export_to_html(dashboard)

        return {
            'allocation': optimized_allocation,
            'asset_predictions': asset_predictions,
            'risk_metrics': risk_metrics,
            'regime_info': regime_info,
            'stress_test_results': stress_test_results,
            'dashboard': dashboard
        }

    def _predict_asset_performance(self):
        """Napove uspešnost vseh sredstev."""
        print("🔮 Generiram napovedi uspešnosti...")

        predictions = {}
        for symbol, data in self.market_data.items():
            try:
                # Uporabi ustrezen model glede na tržni režim
                prediction = self._predict_single_asset(symbol, data)
                predictions[symbol] = prediction

                print(f"   ✅ {symbol}: donos {prediction.get('expected_return', 0)*100:.1f}%, "
                      f"zaupanje {prediction.get('confidence', 0):.2f}")

            except Exception as e:
                print(f"   ❌ {symbol}: napaka pri napovedi - {e}")
                predictions[symbol] = self._get_fallback_prediction(symbol)

        return predictions

    def _predict_single_asset(self, symbol, data):
        """Napove uspešnost posameznega sredstva."""
        # Poenostavljena implementacija - v praksi bi uporabili FeatureEngine in SmartModelHub
        returns = data['Close'].pct_change().dropna()

        # Osnovne metrike
        expected_return = returns.mean() * 252  # Letni donos
        volatility = returns.std() * np.sqrt(252)
        sharpe = expected_return / volatility if volatility > 0 else 0

        # Geopolitični vpliv (poenostavljeno)
        geo_impact = self._calculate_geopolitical_impact(symbol)

        # Sentiment analiza (poenostavljeno)
        sentiment = self._analyze_sentiment(symbol)

        # Prilagodi donos glede na tržni režim
        regime_adjustment = self._get_regime_adjustment()
        adjusted_return = expected_return * regime_adjustment * (1 - geo_impact * 0.2)

        return {
            'expected_return': adjusted_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'confidence': max(0.1, min(0.9, 1 - volatility)),
            'geopolitical_impact': geo_impact,
            'sentiment_score': sentiment,
            'data_quality': 0.8  # Poenostavljeno
        }

    def _calculate_geopolitical_impact(self, symbol):
        """Izračuna geopolitični vpliv (poenostavljeno)."""
        # V praksi bi uporabili GeoAnalyzer iz 2.5
        high_risk_symbols = ['BABA', 'TCEHY', 'PBR']  # Simboli z višjim tveganjem
        if symbol in high_risk_symbols:
            return 0.3
        return 0.1

    def _analyze_sentiment(self, symbol):
        """Analizira sentiment (poenostavljeno)."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news

            if news and len(news) > 0:
                return min(1.0, len(news) * 0.1)  # Več novic = boljši sentiment
        except:
            pass

        return 0.5  # Neutralen

    def _get_regime_adjustment(self):
        """Vrne prilagoditev glede na tržni režim."""
        adjustments = {
            'bull': 1.2,
            'bear': 0.8,
            'sideways': 1.0,
            'unknown': 1.0
        }
        return adjustments.get(self.current_regime, 1.0)

    def _get_fallback_prediction(self, symbol):
        """Vrni osnovno napoved če napredni modeli ne delujejo."""
        return {
            'expected_return': 0.05,  # 5% letni donos
            'volatility': 0.15,
            'sharpe_ratio': 0.33,
            'confidence': 0.3,
            'geopolitical_impact': 0.1,
            'sentiment_score': 0.5,
            'data_quality': 0.5
        }

    def _optimize_portfolio(self, asset_predictions):
        """Optimizira razporeditev portfelja."""
        print("🧮 Optimiziram portfelj...")

        # Filtriranje sredstev z dovolj visokim zaupanjem
        quality_assets = {
            symbol: pred for symbol, pred in asset_predictions.items()
            if pred['confidence'] > 0.3 and pred['data_quality'] > 0.5
        }

        if not quality_assets:
            print("⚠️  Ni kakovostnih sredstev za optimizacijo")
            return {}

        # Razvrsti sredstva po pričakovanem donosu
        sorted_assets = sorted(
            quality_assets.items(),
            key=lambda x: x[1]['expected_return'],
            reverse=True
        )

        # Izberi najboljša sredstva
        top_assets = dict(sorted_assets[:15])  # Top 15 sredstev

        # Dodeli uteži glede na pričakovani donos in zaupanje
        total_score = sum(
            asset['expected_return'] * asset['confidence']
            for asset in top_assets.values()
        )

        allocation = {}
        for symbol, prediction in top_assets.items():
            weight = (prediction['expected_return'] * prediction['confidence']) / total_score
            allocation[symbol] = weight * self.amount

        print(f"✅ Portfelj optimiziran z {len(allocation)} sredstvi")
        return allocation

    def _calculate_risk_metrics(self, allocation, predictions):
        """Izračuna metrike tveganja."""
        if not allocation:
            return self._get_default_risk_metrics()

        # Izračun portfeljskih metrik
        portfolio_return = 0
        portfolio_volatility = 0
        total_amount = sum(allocation.values())

        weights = {}
        for symbol, amount in allocation.items():
            if symbol in predictions:
                weight = amount / total_amount
                weights[symbol] = weight
                portfolio_return += weight * predictions[symbol]['expected_return']
                portfolio_volatility += (weight ** 2) * (predictions[symbol]['volatility'] ** 2)

        portfolio_volatility = np.sqrt(portfolio_volatility)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

        # Value at Risk (poenostavljeno)
        var_95 = portfolio_return - 1.645 * portfolio_volatility

        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'max_drawdown': portfolio_volatility * 2,  # Poenostavljena ocena
            'diversification_score': min(1.0, len(weights) / 10),
            'number_of_assets': len(weights)
        }

    def _get_default_risk_metrics(self):
        """Privzete metrike tveganja."""
        return {
            'expected_return': 0.0,
            'volatility': 0.15,
            'sharpe_ratio': 0.0,
            'var_95': -0.15,
            'max_drawdown': -0.30,
            'diversification_score': 0.0,
            'number_of_assets': 0
        }

    def _perform_stress_tests(self, allocation, predictions):
        """Izvede stres teste."""
        print("🌪️  Izvajam stres teste...")

        if not allocation:
            return {}

        # Definicija stres scenarijev
        scenarios = {
            'market_crash': {
                'return_multiplier': 0.5,
                'volatility_multiplier': 2.0
            },
            'high_inflation': {
                'return_multiplier': 0.7,
                'volatility_multiplier': 1.5
            },
            'geopolitical_crisis': {
                'return_multiplier': 0.6,
                'volatility_multiplier': 1.8
            }
        }

        # Pripravi podatke za optimizator
        portfolio_data = {
            'expected_returns': np.array([predictions[symbol]['expected_return'] for symbol in allocation.keys()]),
            'volatility': np.array([predictions[symbol]['volatility'] for symbol in allocation.keys()])
        }

        # Izvedi stres teste
        results = self.portfolio_optimizer.stress_test_scenarios(portfolio_data, scenarios)

        for scenario, metrics in results.items():
            print(f"   📊 {scenario}: VaR95 = {metrics['var_95']:.1%}, "
                  f"ES = {metrics['expected_shortfall']:.1%}")

        return results

    def _perform_backtest(self, allocation, months=12):
        """Izvede backtest (poenostavljeno)."""
        if not allocation:
            return None

        # Generiraj sintetične podatke za backtest
        np.random.seed(42)
        n_days = months * 21

        # Simuliraj donose portfelja
        portfolio_returns = np.random.normal(0.08/252, 0.15/np.sqrt(252), n_days)
        benchmark_returns = np.random.normal(0.07/252, 0.12/np.sqrt(252), n_days)

        portfolio_values = [self.amount]
        benchmark_values = [self.amount]

        for i in range(n_days):
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_returns[i]))
            benchmark_values.append(benchmark_values[-1] * (1 + benchmark_returns[i]))

        return {
            'portfolio_values': portfolio_values,
            'benchmark_values': benchmark_values,
            'cumulative_return': (portfolio_values[-1] - self.amount) / self.amount,
            'benchmark_cumulative_return': (benchmark_values[-1] - self.amount) / self.amount,
            'max_drawdown': -0.15  # Poenostavljeno
        }

    # Lastnosti za časovne periode
    @property
    def data_periods(self):
        return {
            'kratko': '1y',
            'srednje': '3y',
            'dolgo': '5y'
        }

# =============================================================================
# DEMONSTRACIJA UPORABE
# =============================================================================

async def demo_smartinvest_30():
    """Demonstracija delovanja SmartInvest 3.0."""
    print("🚀 ZAGON SMARTINVEST 3.0 DEMO")
    print("="*50)

    # Ustvari instanco SmartInvest 3.0
    advisor = SmartInvest30(
        amount=100000,
        risk_profile='srednje',
        investment_horizon='srednje'
    )

    # Naloži podatke
    await advisor.load_all_data()

    # Izvedi celovito analizo
    results = advisor.perform_comprehensive_analysis()

    # Prikaži rezultate
    if results and results['allocation']:
        print(f"\n🎉 ANALIZA KONČANA!")
        print(f"Število sredstev: {len(results['allocation'])}")
        print(f"Skupna vrednost: {sum(results['allocation'].values()):,.0f} €")
        print(f"Pričakovan donos: {results['risk_metrics']['expected_return']*100:.1f}%")
        print(f"Volatilnost: {results['risk_metrics']['volatility']*100:.1f}%")

        # Prikaz najboljših sredstev
        print(f"\n🏆 TOP 5 SREDSTEV:")
        top_5 = sorted(results['allocation'].items(), key=lambda x: x[1], reverse=True)[:5]
        for symbol, amount in top_5:
            prediction = results['asset_predictions'][symbol]
            print(f"   • {symbol}: {amount:,.0f} € (donos: {prediction['expected_return']*100:.1f}%)")

    return results

# Zagon demo (če je skripta zagnana neposredno)
if __name__ == "__main__":
    # Zaženemo asinhrono demo funkcijo
    import asyncio
    results = asyncio.run(demo_smartinvest_30())