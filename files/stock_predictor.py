"""
Fast Swing Stock Predictor v4
==============================
Predicts S&P 500 stocks likely to outperform/underperform over 3-5 trading days.
Outputs top 5 BUY + top 5 SHORT picks daily.

v4 Upgrades:
  - Fix 1: Regime filter (trend_up / volatile_down / choppy)
  - Fix 2: News/crash guard (abnormal moves + earnings proximity)
  - Fix 3: Bounce detector (oversold bounce scoring)
  - Fix 4: Asymmetric buy/short thresholds
  - Fix 5: Cross-sectional percentile ranking
  - Fix 6: Finviz sentiment layer

Usage:
    python stock_predictor.py              # Run with defaults
    python stock_predictor.py --train      # Force retrain model
    python stock_predictor.py --top 10     # Show top 10 buys + 10 shorts
    python stock_predictor.py --json       # Save picks to docs/predictions.json
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import logging
import os
import pickle
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_YEARS = 2
FORWARD_DAYS = 5
MIN_VOLUME = 1_000_000
NUM_QUINTILES = 5
MODEL_PATH = "model_lgbm.pkl"

# v4: Regime & Guard Config
REGIME_SPY_THRESH = 0.02        # ±2% 20d SPY return for regime boundary
NEWS_FLAG_RET_THRESH = 0.05     # |1d return| > 5% triggers news flag
NEWS_FLAG_VOL_THRESH = 3.0      # volume > 3x 20d avg triggers news flag
BOUNCE_SCORE_BUY_GATE = 0.6    # bounce_score > 0.6 → force BUY
BOUNCE_SCORE_SHORT_BLOCK = 0.4  # bounce_score > 0.4 → block SHORT
BUY_SCORE_THRESH = 0.60         # asymmetric: buy threshold
SHORT_SCORE_THRESH = 0.25       # asymmetric: short threshold (stricter)
SHORT_MAX_DRAWDOWN = -0.10      # never short stock down >10% in 20 days

# Sentiment keywords
SENTIMENT_POS = [
    "upgrade", "beat", "strong", "buy", "bullish", "growth",
    "raised", "above", "outperform", "positive"
]
SENTIMENT_NEG = [
    "downgrade", "miss", "weak", "sell", "bearish", "decline",
    "cut", "below", "underperform", "negative"
]

# ─────────────────────────────────────────────
# S&P 500 TICKERS
# ─────────────────────────────────────────────
def get_sp500_tickers():
    """Fetch current S&P 500 tickers from Wikipedia."""
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        sectors = dict(zip(
            table["Symbol"].str.replace(".", "-", regex=False),
            table["GICS Sector"]
        ))
        return tickers, sectors
    except Exception as e:
        logging.warning(f"Could not fetch S&P 500 from Wikipedia: {e}. Using fallback list.")
        # Fallback: top 50 liquid stocks
        fallback = [
            "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK-B","UNH","JNJ",
            "JPM","V","PG","XOM","HD","MA","CVX","MRK","ABBV","PEP",
            "KO","COST","AVGO","LLY","WMT","MCD","CSCO","TMO","ABT","DHR",
            "ACN","NEE","LIN","TXN","PM","UNP","RTX","LOW","AMGN","HON",
            "IBM","CAT","DE","GS","BA","AXP","SBUX","MDLZ","BLK","ADI"
        ]
        return fallback, {}

# ─────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────
def download_data(tickers, years=TRAIN_YEARS):
    """Download OHLCV data for all tickers."""
    end = datetime.today()
    start = end - timedelta(days=years * 365 + 60)  # extra buffer for feature calc

    print(f"Downloading {len(tickers)} tickers ({years}y of data)...")
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True, threads=True)
    print("Download complete.")
    return data

# ─────────────────────────────────────────────
# FIX 1: REGIME DETECTION
# ─────────────────────────────────────────────
def compute_regime(spy_data):
    """Detect market regime from SPY price action.

    Returns:
        (regime_str, regime_int): e.g. ("volatile_down", -1)
    """
    try:
        close = spy_data['Close'].dropna()
        if len(close) < 22:
            return "choppy", 0
        ret_20d = close.iloc[-1] / close.iloc[-21] - 1
        vol_20d = close.pct_change().tail(20).std()
        if ret_20d > REGIME_SPY_THRESH and vol_20d < 0.012:
            return "trend_up", 1
        elif ret_20d < -REGIME_SPY_THRESH and vol_20d > 0.012:
            return "volatile_down", -1
        else:
            return "choppy", 0
    except Exception:
        return "choppy", 0

# ─────────────────────────────────────────────
# FIX 2: NEWS / CRASH GUARD
# ─────────────────────────────────────────────
def compute_news_flag(ticker, df):
    """Check if stock had a news-driven abnormal move in the last 5 days,
    or has earnings within the next 7 days.

    Returns:
        (flagged: bool, reason: str)
    """
    if len(df) < 25:
        return False, ""

    close = df['Close']
    volume = df['Volume']
    vol_20_avg = volume.rolling(20).mean()

    # Check last 5 days for abnormal price + volume move
    for i in range(-5, 0):
        try:
            ret = close.iloc[i] / close.iloc[i - 1] - 1
            avg_vol = vol_20_avg.iloc[i]
            vol_spike = volume.iloc[i] / avg_vol if avg_vol > 0 else 0
            if abs(ret) > NEWS_FLAG_RET_THRESH and vol_spike > NEWS_FLAG_VOL_THRESH:
                return True, f"{ret * 100:.1f}% on {vol_spike:.1f}x volume"
        except Exception:
            continue

    # Best-effort earnings check (yfinance calendar is unreliable; wrap in try/except)
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is not None and not cal.empty:
            earnings_date = None
            if hasattr(cal, 'columns') and 'Earnings Date' in cal.columns:
                earnings_date = pd.to_datetime(cal['Earnings Date'].iloc[0])
            elif hasattr(cal, 'index') and 'Earnings Date' in cal.index:
                earnings_date = pd.to_datetime(cal.loc['Earnings Date'].iloc[0])
            if earnings_date is not None:
                days_to = (earnings_date.tz_localize(None) - datetime.today()).days
                if 0 <= days_to <= 7:
                    return True, f"Earnings in {days_to} days"
    except Exception:
        pass

    return False, ""

# ─────────────────────────────────────────────
# FIX 3: BOUNCE DETECTOR
# ─────────────────────────────────────────────
def compute_bounce_score(feat_row):
    """Score oversold bounce potential (0-1 scale).

    Higher score = stronger oversold bounce candidate → favor BUY, block SHORT.
    """
    score = 0.0
    if feat_row.get('rsi_5', 50) < 25:
        score += 0.30
    if feat_row.get('ret_20d', 0) < -0.15:
        score += 0.20
    if feat_row.get('vol_spike', 1) > 2.0:
        score += 0.15
    if feat_row.get('close_in_range', 0.5) > 0.5:
        score += 0.20
    if feat_row.get('sector_breadth', 0.5) > 0.5:
        score += 0.15
    return min(score, 1.0)

# ─────────────────────────────────────────────
# FIX 6: SENTIMENT LAYER
# ─────────────────────────────────────────────
def fetch_sentiment(ticker):
    """Scrape Finviz headlines for a ticker and return sentiment score (-1 to +1).

    Returns 0.0 on any failure (timeout, 403, parse error, etc.).
    """
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}&ty=c&ta=1&p=d"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; swing-predictor/4.0)"}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200:
            return 0.0
        soup = BeautifulSoup(resp.text, "html.parser")
        news_table = soup.find(id="news-table")
        if not news_table:
            return 0.0
        headlines = [a.get_text().lower() for a in news_table.find_all("a")]
        text = " ".join(headlines)
        pos = sum(text.count(w) for w in SENTIMENT_POS)
        neg = sum(text.count(w) for w in SENTIMENT_NEG)
        return float(max(-1.0, min(1.0, (pos - neg) / (pos + neg + 1))))
    except Exception:
        return 0.0

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def compute_features(df):
    """Compute all features for a single stock DataFrame (OHLCV)."""
    f = pd.DataFrame(index=df.index)
    c, h, l, o, v = df['Close'], df['High'], df['Low'], df['Open'], df['Volume']

    # --- A: Momentum (7) ---
    f['ret_1d'] = c.pct_change(1)
    f['ret_3d'] = c.pct_change(3)
    f['ret_5d'] = c.pct_change(5)
    f['ret_10d'] = c.pct_change(10)
    f['ret_20d'] = c.pct_change(20)   # NEW: needed for short guard + bounce score
    f['accel_5v20'] = c.pct_change(5) - c.pct_change(20)
    f['gap'] = o / c.shift(1) - 1

    # --- B: Mean Reversion (5) ---
    f['rsi_5'] = RSIIndicator(c, window=5).rsi()
    f['rsi_14'] = RSIIndicator(c, window=14).rsi()
    bb = BollingerBands(c, window=20, window_dev=2)
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()
    f['bb_position'] = (c - bb_low) / (bb_high - bb_low + 1e-10)
    f['dist_5d_high'] = c / c.rolling(5).max() - 1
    f['dist_20d_high'] = c / c.rolling(20).max() - 1

    # --- C: Volume (5) ---
    vol_20 = v.rolling(20).mean().replace(0, np.nan)
    f['vol_ratio_5_20'] = v.rolling(5).mean() / vol_20
    f['vol_spike'] = v / vol_20
    f['obv_slope'] = OnBalanceVolumeIndicator(c, v).on_balance_volume().diff(5)
    f['vol_price_div'] = f['ret_5d'] * np.log1p(f['vol_ratio_5_20'])
    typical_price = (h + l + c) / 3
    vol_20_sum = v.rolling(20).sum().replace(0, np.nan)
    vwap = (typical_price * v).rolling(20).sum() / vol_20_sum
    f['vwap_dist'] = c / vwap - 1

    # --- D: Volatility (4) ---
    atr = AverageTrueRange(h, l, c, window=14)
    f['atr_pct'] = atr.average_true_range() / (c + 1e-10)
    f['intraday_range'] = ((h - l) / (c + 1e-10)).rolling(5).mean()
    atr5 = AverageTrueRange(h, l, c, window=5).average_true_range()
    atr20 = AverageTrueRange(h, l, c, window=20).average_true_range()
    f['vol_compression'] = atr5 / (atr20 + 1e-10)
    f['close_in_range'] = (c - l) / (h - l + 1e-10)

    # --- E: Sector features added later (cross-sectional) ---

    # --- F: Context features — placeholders, overwritten in predict_today ---
    f['regime'] = 0
    f['bounce_score'] = 0.0
    f['news_flag'] = 0
    f['sentiment_score'] = 0.0

    return f


def add_sector_features_fast(all_features, sectors):
    """Vectorized sector feature computation — much faster than row-by-row."""
    tickers = list(all_features.keys())
    if not tickers:
        return all_features

    ret_3d_df = pd.DataFrame({t: all_features[t]['ret_3d'] for t in tickers if 'ret_3d' in all_features[t].columns})
    ret_5d_df = pd.DataFrame({t: all_features[t]['ret_5d'] for t in tickers if 'ret_5d' in all_features[t].columns})

    spy_ret_3d = ret_3d_df.mean(axis=1)

    sector_map = pd.Series({t: sectors.get(t, 'Unknown') for t in tickers})
    unique_sectors = sector_map.unique()

    for sector in unique_sectors:
        sector_tickers = sector_map[sector_map == sector].index.tolist()
        sector_tickers = [t for t in sector_tickers if t in ret_3d_df.columns]
        if not sector_tickers:
            continue

        sector_ret_3d = ret_3d_df[sector_tickers].mean(axis=1)
        sector_breadth = (ret_3d_df[sector_tickers] > 0).mean(axis=1)
        sector_vs_mkt = sector_ret_3d - spy_ret_3d
        sector_ret_5d_rank = ret_5d_df[sector_tickers].rank(axis=1, pct=True)

        for t in sector_tickers:
            if t not in all_features:
                continue
            feat = all_features[t]
            feat['sector_ret_3d'] = sector_ret_3d
            feat['sector_breadth'] = sector_breadth
            feat['rank_in_sector'] = sector_ret_5d_rank[t] if t in sector_ret_5d_rank.columns else 0.5
            feat['spy_ret_3d'] = spy_ret_3d
            feat['sector_vs_market'] = sector_vs_mkt

    return all_features


def add_bounce_scores(all_features):
    """Compute vectorized bounce_score for training data (after sector features available)."""
    for ticker, feat in all_features.items():
        scores = pd.Series(0.0, index=feat.index)
        if 'rsi_5' in feat.columns:
            scores = scores + (feat['rsi_5'] < 25).astype(float) * 0.30
        if 'ret_20d' in feat.columns:
            scores = scores + (feat['ret_20d'] < -0.15).astype(float) * 0.20
        if 'vol_spike' in feat.columns:
            scores = scores + (feat['vol_spike'] > 2.0).astype(float) * 0.15
        if 'close_in_range' in feat.columns:
            scores = scores + (feat['close_in_range'] > 0.5).astype(float) * 0.20
        if 'sector_breadth' in feat.columns:
            scores = scores + (feat['sector_breadth'] > 0.5).astype(float) * 0.15
        feat['bounce_score'] = scores.clip(0, 1)
    return all_features


# ─────────────────────────────────────────────
# TARGET VARIABLE
# ─────────────────────────────────────────────
def compute_target(close, forward_days=FORWARD_DAYS):
    """Compute forward return."""
    return close.shift(-forward_days) / close - 1

# ─────────────────────────────────────────────
# BUILD TRAINING DATA
# ─────────────────────────────────────────────
FEATURE_COLS = [
    # Momentum (7)
    'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d', 'accel_5v20', 'gap',
    # Mean Reversion (5)
    'rsi_5', 'rsi_14', 'bb_position', 'dist_5d_high', 'dist_20d_high',
    # Volume (5)
    'vol_ratio_5_20', 'vol_spike', 'obv_slope', 'vol_price_div', 'vwap_dist',
    # Volatility (4)
    'atr_pct', 'intraday_range', 'vol_compression', 'close_in_range',
    # Sector/Cross-sectional (5)
    'sector_ret_3d', 'sector_breadth', 'rank_in_sector', 'spy_ret_3d', 'sector_vs_market',
    # Context features (4) — placeholders 0 in training; real values at prediction time
    'regime', 'bounce_score', 'news_flag', 'sentiment_score',
]

def build_dataset(data, tickers, sectors):
    """Process all tickers and build one big training DataFrame."""
    all_features = {}
    close_dict = {}

    print("Computing features...")
    processed = 0
    for ticker in tickers:
        try:
            if len(tickers) > 1:
                df = data[ticker].dropna(subset=['Close'])
            else:
                df = data.dropna(subset=['Close'])

            if len(df) < 100:
                continue

            if df['Volume'].mean() < MIN_VOLUME:
                continue

            feat = compute_features(df)
            all_features[ticker] = feat
            close_dict[ticker] = df['Close']
            processed += 1
        except Exception as e:
            logging.warning(f"Skipped {ticker}: {e}")
            continue

    print(f"Computed features for {processed} stocks")

    # Add sector features
    print("Adding sector features...")
    all_features = add_sector_features_fast(all_features, sectors)

    # Add bounce scores (vectorized, now that sector_breadth is available)
    print("Computing bounce scores...")
    all_features = add_bounce_scores(all_features)

    # Combine into training set
    rows = []
    for ticker, feat in all_features.items():
        if ticker not in close_dict:
            continue
        fwd = compute_target(close_dict[ticker])
        feat = feat.copy()
        feat['forward_return'] = fwd
        feat['ticker'] = ticker
        feat['date'] = feat.index
        rows.append(feat)

    if not rows:
        raise ValueError("No data processed. Check tickers/connection.")

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.sort_values('date').reset_index(drop=True)

    # Compute quintiles per date (cross-sectional)
    combined['quintile'] = combined.groupby('date')['forward_return'].transform(
        lambda x: pd.qcut(x, NUM_QUINTILES, labels=False, duplicates='drop')
    )

    combined = combined.dropna(subset=FEATURE_COLS + ['quintile'])
    print(f"Training set: {len(combined)} rows")

    return combined, all_features, close_dict

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def train_model(combined):
    """Train LightGBM on the dataset."""
    X = combined[FEATURE_COLS].values
    y = combined['quintile'].astype(int).values

    split = int(len(combined) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = LGBMClassifier(
        objective='multiclass',
        num_class=NUM_QUINTILES,
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=100,
        reg_alpha=0.1,
        reg_lambda=1.0,
        verbose=-1,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    accuracy = (val_pred == y_val).mean()
    top_mask = val_pred >= (NUM_QUINTILES - 2)
    if top_mask.sum() > 0:
        top_actual = y_val[top_mask]
        hit_rate = (top_actual >= (NUM_QUINTILES - 2)).mean()
    else:
        hit_rate = 0

    print(f"\n--- Validation ---")
    print(f"Overall accuracy: {accuracy:.1%}")
    print(f"Top-pick hit rate (pred Q4-5, actual Q4-5): {hit_rate:.1%}")

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\nTop 10 features:")
    for feat, imp in importance.head(10).items():
        print(f"  {feat:25s} {imp:6.0f}")

    return model

# ─────────────────────────────────────────────
# MOMENTUM BASELINE SCORE
# ─────────────────────────────────────────────
def momentum_score(feat_row):
    """Simple weighted momentum composite (0-1 scale)."""
    scores = []
    scores.append(feat_row.get('ret_3d', 0) * 0.3)
    scores.append(min(feat_row.get('vol_spike', 1), 3) / 3 * 0.2)
    rsi = feat_row.get('rsi_5', 50)
    scores.append((100 - rsi) / 100 * 0.2)
    scores.append(feat_row.get('sector_breadth', 0.5) * 0.15)
    scores.append(feat_row.get('rank_in_sector', 0.5) * 0.15)
    return sum(scores)

# ─────────────────────────────────────────────
# PREDICTION (v4)
# ─────────────────────────────────────────────
def predict_today(model, all_features, sectors, data=None, tickers=None,
                  regime_str="choppy", regime_int=0):
    """Generate today's picks with regime filtering, news guard, and asymmetric thresholds.

    Returns:
        (results_df, buys_df, shorts_df, blocked_list, regime_str)
    """
    results = []
    blocked = []

    # Core features that must be non-NaN (context features set dynamically)
    core_cols = [c for c in FEATURE_COLS if c not in ('regime', 'bounce_score', 'news_flag', 'sentiment_score')]

    multi_ticker = tickers and len(tickers) > 1

    for ticker, feat in all_features.items():
        if feat.empty:
            continue

        latest = feat.iloc[-1].copy()

        # Require all core features to be present
        if any(pd.isna(latest.get(col, np.nan)) for col in core_cols):
            continue

        # Fix 2: News / crash guard
        if data is not None:
            try:
                df = data[ticker].dropna(subset=['Close']) if multi_ticker else data.dropna(subset=['Close'])
                flagged, flag_reason = compute_news_flag(ticker, df)
                if flagged:
                    blocked.append({'ticker': ticker, 'reason': flag_reason})
                    continue
            except Exception:
                pass

        # Fix 1 & 3: Set context features on this row
        bounce_score = compute_bounce_score(latest)
        latest['regime'] = regime_int
        latest['bounce_score'] = bounce_score
        latest['news_flag'] = 0
        latest['sentiment_score'] = 0.0

        X = np.array([[latest.get(col, 0) for col in FEATURE_COLS]])

        # LightGBM prediction
        proba = model.predict_proba(X)[0]
        n_classes = len(proba)
        lgbm_score = sum(proba[i] * i for i in range(n_classes)) / max(n_classes - 1, 1)

        # Momentum baseline
        mom_score = momentum_score(latest)

        # Fix ensemble: 55% LightGBM + 25% momentum + 20% bounce
        final_score = 0.55 * lgbm_score + 0.25 * mom_score + 0.20 * bounce_score

        # Signal reasons
        signals = []
        if latest.get('vol_spike', 0) > 1.5:
            signals.append("Vol spike")
        if latest.get('rsi_5', 50) < 30:
            signals.append("RSI oversold")
        if latest.get('rsi_5', 50) > 70:
            signals.append("RSI overbought")
        if latest.get('vol_compression', 1) < 0.7:
            signals.append("Vol squeeze")
        if latest.get('sector_breadth', 0.5) > 0.7:
            signals.append("Sector strong")
        if latest.get('sector_breadth', 0.5) < 0.3:
            signals.append("Sector weak")
        if latest.get('ret_3d', 0) > 0.03:
            signals.append("Strong momentum")
        if latest.get('ret_3d', 0) < -0.03:
            signals.append("Sharp pullback")
        if latest.get('bb_position', 0.5) < 0.1:
            signals.append("Near BB low")
        if bounce_score > 0.6:
            signals.append("Oversold bounce")

        sector = sectors.get(ticker, '?')
        signal_str = " + ".join(signals[:3]) if signals else "Mixed signals"

        results.append({
            'ticker': ticker,
            'score': final_score,
            'lgbm': lgbm_score,
            'momentum': mom_score,
            'bounce_score': bounce_score,
            'sector': sector,
            'signal': signal_str,
            'ret_3d': float(latest.get('ret_3d', 0)),
            'ret_20d': float(latest.get('ret_20d', 0)),
            'rsi_5': float(latest.get('rsi_5', 50)),
            'vol_spike': float(latest.get('vol_spike', 1)),
            'sentiment_score': 0.0,
        })

    if not results:
        empty = pd.DataFrame()
        return empty, empty, empty, blocked, regime_str

    results_df = pd.DataFrame(results)

    # Fix 5: Cross-sectional percentile rank
    results_df['cross_rank'] = results_df['score'].rank(pct=True)

    # Fix 4 & 5: Asymmetric candidate selection
    # Buy candidates: top 10% cross-rank
    buy_candidates = results_df[results_df['cross_rank'] >= 0.90].copy()
    buy_candidates = buy_candidates[
        (buy_candidates['score'] > BUY_SCORE_THRESH) |
        (buy_candidates['bounce_score'] > BOUNCE_SCORE_BUY_GATE)
    ]

    # Short candidates: bottom 10% cross-rank with strict guards
    short_candidates = results_df[results_df['cross_rank'] <= 0.10].copy()
    short_candidates = short_candidates[
        (short_candidates['score'] < SHORT_SCORE_THRESH) &       # stricter threshold
        (short_candidates['ret_20d'] > SHORT_MAX_DRAWDOWN) &     # not already crashed
        (short_candidates['bounce_score'] <= BOUNCE_SCORE_SHORT_BLOCK)  # no bounce potential
    ]

    # Fix 1: Regime gate — volatile_down caps shorts at 3
    if regime_str == "volatile_down":
        short_candidates = short_candidates.sort_values('score').head(3)

    # Fix 6: Fetch sentiment for top candidates only (avoid scraping all 500)
    candidate_tickers = (
        list(buy_candidates.head(20)['ticker']) +
        list(short_candidates.head(20)['ticker'])
    )
    sentiment_tickers = list(set(candidate_tickers))

    if sentiment_tickers:
        print(f"\nFetching sentiment for {len(sentiment_tickers)} candidates...")
        sentiment_map = {}
        for t in sentiment_tickers:
            sentiment_map[t] = fetch_sentiment(t)
            time.sleep(1)
        buy_candidates = buy_candidates.copy()
        short_candidates = short_candidates.copy()
        buy_candidates['sentiment_score'] = buy_candidates['ticker'].map(sentiment_map).fillna(0.0)
        short_candidates['sentiment_score'] = short_candidates['ticker'].map(sentiment_map).fillna(0.0)

    buys = buy_candidates.sort_values('score', ascending=False).reset_index(drop=True)
    shorts = short_candidates.sort_values('score', ascending=True).reset_index(drop=True)
    results_df = results_df.sort_values('score', ascending=False).reset_index(drop=True)

    return results_df, buys, shorts, blocked, regime_str

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
_REGIME_DESC = {
    "trend_up":      "trending up — favoring momentum buys",
    "volatile_down": "volatile/falling — favoring mean reversion, shorts restricted to 3",
    "choppy":        "choppy — being selective",
}

def print_picks(buys, shorts, blocked, regime_str, results_df, top_n=5):
    """Print formatted daily picks with regime, bounce, sentiment, and blocked stocks."""
    today = datetime.today().strftime('%Y-%m-%d')
    n_analyzed = len(results_df) if results_df is not None and not results_df.empty else 0
    regime_desc = _REGIME_DESC.get(regime_str, regime_str)

    print("\n" + "=" * 95)
    print(f"  DAILY PICKS — {today}")
    print(f"  Stocks analyzed: {n_analyzed} | Model: LightGBM + Momentum + Bounce Ensemble v4")
    print(f"  REGIME: {regime_str.upper()} — {regime_desc}")
    print("=" * 95)

    col_hdr = f"  {'Rank':>4} | {'Ticker':<6} | {'Score':>6} | {'Bounce':>6} | {'Sent':>5} | {'Conf':<4} | {'Sector':<22} | Signal"
    sep = "  " + "-" * 91

    # TOP BUYS
    top_buys = buys.head(top_n)
    print(f"\n  🟢 TOP {top_n} BUYS (hold 3-5 days):")
    print(sep)
    print(col_hdr)
    print(sep)
    for i, (_, row) in enumerate(top_buys.iterrows(), 1):
        conf = "HIGH" if row['score'] > 0.65 else ("MED" if row['score'] > 0.55 else "LOW")
        sent = f"{row.get('sentiment_score', 0):+.2f}"
        print(f"  {i:>4} | {row['ticker']:<6} | {row['score']:.3f} | {row['bounce_score']:.3f}  | {sent:>5} | {conf:<4} | {row['sector']:<22} | {row['signal']}")

    # TOP SHORTS
    top_shorts = shorts.head(top_n)
    print(f"\n  🔴 TOP {top_n} SHORTS (avoid / short 3-5 days):")
    print(sep)
    print(col_hdr)
    print(sep)
    for i, (_, row) in enumerate(top_shorts.iterrows(), 1):
        conf = "HIGH" if row['score'] < 0.35 else ("MED" if row['score'] < 0.45 else "LOW")
        sent = f"{row.get('sentiment_score', 0):+.2f}"
        print(f"  {i:>4} | {row['ticker']:<6} | {row['score']:.3f} | {row['bounce_score']:.3f}  | {sent:>5} | {conf:<4} | {row['sector']:<22} | {row['signal']}")

    # BLOCKED
    if blocked:
        print(f"\n  ⛔ BLOCKED (news-driven / earnings proximity — skipped):")
        print("  " + "-" * 60)
        for b in blocked:
            print(f"     {b['ticker']:<8} — flagged: {b['reason']}")

    print("\n" + "=" * 95)
    print("  Score: 0-1  |  Buy threshold: >0.60  |  Short threshold: <0.25")
    print("  Bounce: oversold bounce probability  |  Sent: Finviz headline sentiment")
    print("=" * 95 + "\n")

    return top_buys, top_shorts


def save_json(buys, shorts, n_analyzed, regime, blocked, output_path="docs/predictions.json"):
    """Save today's picks as JSON for the GitHub Pages dashboard."""
    def row_to_dict(rank, row, is_buy):
        score = float(row['score'])
        if is_buy:
            conf = "HIGH" if score > 0.65 else ("MED" if score > 0.55 else "LOW")
        else:
            conf = "HIGH" if score < 0.35 else ("MED" if score < 0.45 else "LOW")
        return {
            "rank": rank,
            "ticker": str(row['ticker']),
            "score": round(score, 4),
            "bounce_score": round(float(row.get('bounce_score', 0)), 4),
            "sentiment_score": round(float(row.get('sentiment_score', 0)), 4),
            "confidence": conf,
            "sector": str(row['sector']),
            "signal": str(row['signal']),
            "ret_3d": round(float(row['ret_3d']), 4),
            "ret_20d": round(float(row.get('ret_20d', 0)), 4),
            "rsi_5": round(float(row['rsi_5']), 1),
            "vol_spike": round(float(row['vol_spike']), 2),
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "date": datetime.today().strftime('%Y-%m-%d'),
        "stocks_analyzed": int(n_analyzed),
        "regime": regime,
        "buys": [row_to_dict(i, row, True) for i, (_, row) in enumerate(buys.iterrows(), 1)],
        "shorts": [row_to_dict(i, row, False) for i, (_, row) in enumerate(shorts.iterrows(), 1)],
        "blocked": blocked,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Predictions JSON saved to {output_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fast Swing Stock Predictor v4")
    parser.add_argument('--train', action='store_true', help='Force retrain model')
    parser.add_argument('--top', type=int, default=5, help='Number of top picks (default 5)')
    parser.add_argument('--years', type=int, default=TRAIN_YEARS, help='Years of training data')
    parser.add_argument('--json', action='store_true', help='Also save picks to docs/predictions.json')
    args = parser.parse_args()

    # 1. Get tickers
    print("Fetching S&P 500 tickers...")
    tickers, sectors = get_sp500_tickers()
    print(f"Found {len(tickers)} tickers")

    # 2. Download stock data
    data = download_data(tickers, years=args.years)

    # 3. Detect market regime from SPY
    print("Detecting market regime...")
    regime_str, regime_int = "choppy", 0
    try:
        end = datetime.today()
        start = end - timedelta(days=60)
        spy_data = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        regime_str, regime_int = compute_regime(spy_data)
        print(f"Regime: {regime_str.upper()}")
    except Exception as e:
        logging.warning(f"Could not detect regime from SPY: {e}. Defaulting to choppy.")

    # 4. Build features + dataset
    combined, all_features, close_dict = build_dataset(data, tickers, sectors)

    # 5. Train or load model
    if args.train or not os.path.exists(MODEL_PATH):
        print("\nTraining model...")
        model = train_model(combined)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {MODEL_PATH}")
    else:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Loaded saved model")

    # 6. Generate today's picks
    print("\nGenerating predictions...")
    results_df, buys, shorts, blocked, regime_str = predict_today(
        model, all_features, sectors,
        data=data, tickers=tickers,
        regime_str=regime_str, regime_int=regime_int,
    )

    # 7. Output
    top_buys, top_shorts = print_picks(buys, shorts, blocked, regime_str, results_df, top_n=args.top)

    # 8. Save full rankings to CSV
    results_df.to_csv("predictions.csv", index=False)
    print("Full rankings saved to predictions.csv")

    # 9. Save JSON for GitHub Pages dashboard
    if args.json:
        save_json(top_buys, top_shorts, n_analyzed=len(results_df),
                  regime=regime_str, blocked=blocked)

if __name__ == "__main__":
    main()
