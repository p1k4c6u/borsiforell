"""
Fast Swing Stock Predictor
==========================
Predicts S&P 500 stocks likely to outperform/underperform over 3-5 trading days.
Outputs top 5 BUY + top 5 SHORT picks daily.

Usage:
    python stock_predictor.py              # Run with defaults
    python stock_predictor.py --train      # Force retrain model
    python stock_predictor.py --top 10     # Show top 10 buys + 10 shorts
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_YEARS = 2
FORWARD_DAYS = 5
MIN_VOLUME = 1_000_000
NUM_QUINTILES = 5
MODEL_PATH = "model_lgbm.pkl"

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
    except Exception:
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
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def compute_features(df):
    """Compute all 25 features for a single stock DataFrame (OHLCV)."""
    f = pd.DataFrame(index=df.index)
    c, h, l, o, v = df['Close'], df['High'], df['Low'], df['Open'], df['Volume']

    # --- A: Momentum (6) ---
    f['ret_1d'] = c.pct_change(1)
    f['ret_3d'] = c.pct_change(3)
    f['ret_5d'] = c.pct_change(5)
    f['ret_10d'] = c.pct_change(10)
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
    vol_20 = v.rolling(20).mean()
    f['vol_ratio_5_20'] = v.rolling(5).mean() / (vol_20 + 1)
    f['vol_spike'] = v / (vol_20 + 1)
    f['obv_slope'] = OnBalanceVolumeIndicator(c, v).on_balance_volume().diff(5)
    # Volume-price divergence: price up but volume down = bearish divergence
    f['vol_price_div'] = f['ret_5d'] * np.log1p(f['vol_ratio_5_20'])
    # VWAP distance (approx using typical price * volume)
    typical_price = (h + l + c) / 3
    vwap = (typical_price * v).rolling(20).sum() / (v.rolling(20).sum() + 1)
    f['vwap_dist'] = c / vwap - 1

    # --- D: Volatility (4) ---
    atr = AverageTrueRange(h, l, c, window=14)
    f['atr_pct'] = atr.average_true_range() / (c + 1e-10)
    f['intraday_range'] = ((h - l) / (c + 1e-10)).rolling(5).mean()
    atr5 = AverageTrueRange(h, l, c, window=5).average_true_range()
    atr20 = AverageTrueRange(h, l, c, window=20).average_true_range()
    f['vol_compression'] = atr5 / (atr20 + 1e-10)
    f['close_in_range'] = (c - l) / (h - l + 1e-10)

    # --- E: Sector features are added later (cross-sectional) ---

    return f

def add_sector_features(all_features, sectors):
    """Add cross-sectional sector features across all stocks."""
    combined = []

    for date in all_features[list(all_features.keys())[0]].index:
        row_data = {}
        for ticker, feat_df in all_features.items():
            if date in feat_df.index:
                row_data[ticker] = feat_df.loc[date]

        if len(row_data) < 20:
            continue

        cross = pd.DataFrame(row_data).T
        cross['sector'] = cross.index.map(lambda t: sectors.get(t, 'Unknown'))

        # SPY/market return (average of all stocks as proxy)
        spy_ret_3d = cross['ret_3d'].mean()

        for ticker in cross.index:
            sector = cross.loc[ticker, 'sector']
            sector_mask = cross['sector'] == sector
            sector_stocks = cross[sector_mask]

            feat = all_features[ticker]
            if date in feat.index:
                feat.loc[date, 'sector_ret_3d'] = sector_stocks['ret_3d'].mean()
                feat.loc[date, 'sector_breadth'] = (sector_stocks['ret_3d'] > 0).mean()
                feat.loc[date, 'rank_in_sector'] = sector_stocks['ret_5d'].rank(pct=True).get(ticker, 0.5)
                feat.loc[date, 'spy_ret_3d'] = spy_ret_3d
                feat.loc[date, 'sector_vs_market'] = sector_stocks['ret_3d'].mean() - spy_ret_3d

    return all_features

def add_sector_features_fast(all_features, sectors):
    """Vectorized sector feature computation — much faster than row-by-row."""
    # Build a cross-sectional DataFrame of key metrics
    tickers = list(all_features.keys())
    if not tickers:
        return all_features

    sample_index = all_features[tickers[0]].index

    # Gather ret_3d and ret_5d for all tickers
    ret_3d_df = pd.DataFrame({t: all_features[t]['ret_3d'] for t in tickers if 'ret_3d' in all_features[t].columns})
    ret_5d_df = pd.DataFrame({t: all_features[t]['ret_5d'] for t in tickers if 'ret_5d' in all_features[t].columns})

    # Market-level
    spy_ret_3d = ret_3d_df.mean(axis=1)

    # Sector mapping
    sector_map = pd.Series({t: sectors.get(t, 'Unknown') for t in tickers})
    unique_sectors = sector_map.unique()

    # Pre-compute sector averages
    for sector in unique_sectors:
        sector_tickers = sector_map[sector_map == sector].index.tolist()
        sector_tickers = [t for t in sector_tickers if t in ret_3d_df.columns]
        if not sector_tickers:
            continue

        sector_ret_3d = ret_3d_df[sector_tickers].mean(axis=1)
        sector_breadth = (ret_3d_df[sector_tickers] > 0).mean(axis=1)
        sector_vs_mkt = sector_ret_3d - spy_ret_3d

        # Rank within sector
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
    'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'accel_5v20', 'gap',
    'rsi_5', 'rsi_14', 'bb_position', 'dist_5d_high', 'dist_20d_high',
    'vol_ratio_5_20', 'vol_spike', 'obv_slope', 'vol_price_div', 'vwap_dist',
    'atr_pct', 'intraday_range', 'vol_compression', 'close_in_range',
    'sector_ret_3d', 'sector_breadth', 'rank_in_sector', 'spy_ret_3d', 'sector_vs_market'
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

            # Filter low volume
            if df['Volume'].mean() < MIN_VOLUME:
                continue

            feat = compute_features(df)
            all_features[ticker] = feat
            close_dict[ticker] = df['Close']
            processed += 1
        except Exception:
            continue

    print(f"Computed features for {processed} stocks")

    # Add sector features
    print("Adding sector features...")
    all_features = add_sector_features_fast(all_features, sectors)

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

    # Use last 20% as validation
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

    # Quick validation
    val_pred = model.predict(X_val)
    accuracy = (val_pred == y_val).mean()
    # Top quintile hit rate
    top_mask = val_pred >= (NUM_QUINTILES - 2)
    if top_mask.sum() > 0:
        top_actual = y_val[top_mask]
        hit_rate = (top_actual >= (NUM_QUINTILES - 2)).mean()
    else:
        hit_rate = 0

    print(f"\n--- Validation ---")
    print(f"Overall accuracy: {accuracy:.1%}")
    print(f"Top-pick hit rate (pred Q4-5, actual Q4-5): {hit_rate:.1%}")

    # Feature importance
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
    # Higher recent return = bullish
    scores.append(feat_row.get('ret_3d', 0) * 0.3)
    # Volume spike = attention
    scores.append(min(feat_row.get('vol_spike', 1), 3) / 3 * 0.2)
    # Low RSI = oversold bounce potential
    rsi = feat_row.get('rsi_5', 50)
    scores.append((100 - rsi) / 100 * 0.2)
    # Sector breadth
    scores.append(feat_row.get('sector_breadth', 0.5) * 0.15)
    # Rank in sector
    scores.append(feat_row.get('rank_in_sector', 0.5) * 0.15)
    return sum(scores)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_today(model, all_features, sectors):
    """Generate today's picks."""
    results = []

    for ticker, feat in all_features.items():
        if feat.empty:
            continue

        latest = feat.iloc[-1]

        # Check we have all features
        if any(pd.isna(latest.get(col, np.nan)) for col in FEATURE_COLS):
            continue

        X = np.array([[latest[col] for col in FEATURE_COLS]])

        # LightGBM prediction (probabilities)
        proba = model.predict_proba(X)[0]
        lgbm_score = sum(proba[i] * i for i in range(len(proba))) / (NUM_QUINTILES - 1)

        # Momentum baseline
        mom_score = momentum_score(latest)

        # Ensemble
        final_score = 0.6 * lgbm_score + 0.4 * mom_score

        # Primary signal reason
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

        sector = sectors.get(ticker, '?')
        signal_str = " + ".join(signals[:3]) if signals else "Mixed signals"

        results.append({
            'ticker': ticker,
            'score': final_score,
            'lgbm': lgbm_score,
            'momentum': mom_score,
            'sector': sector,
            'signal': signal_str,
            'ret_3d': latest.get('ret_3d', 0),
            'rsi_5': latest.get('rsi_5', 50),
            'vol_spike': latest.get('vol_spike', 1),
        })

    return pd.DataFrame(results).sort_values('score', ascending=False)

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
def print_picks(results_df, top_n=5):
    """Print formatted daily picks."""
    today = datetime.today().strftime('%Y-%m-%d')

    print("\n" + "=" * 80)
    print(f"  DAILY PICKS — {today}")
    print(f"  Stocks analyzed: {len(results_df)} | Model: LightGBM + Momentum Ensemble")
    print("=" * 80)

    # TOP BUYS
    print(f"\n  🟢 TOP {top_n} BUYS (hold 3-5 days):")
    print("  " + "-" * 76)
    print(f"  {'Rank':>4} | {'Ticker':<6} | {'Score':>6} | {'Conf':<6} | {'Sector':<16} | Signal")
    print("  " + "-" * 76)

    buys = results_df.head(top_n)
    for i, (_, row) in enumerate(buys.iterrows(), 1):
        conf = "HIGH" if row['score'] > 0.65 else ("MED" if row['score'] > 0.55 else "LOW")
        print(f"  {i:>4} | {row['ticker']:<6} | {row['score']:.3f} | {conf:<6} | {row['sector']:<16} | {row['signal']}")

    # TOP SHORTS
    print(f"\n  🔴 TOP {top_n} SHORTS (avoid / short 3-5 days):")
    print("  " + "-" * 76)
    print(f"  {'Rank':>4} | {'Ticker':<6} | {'Score':>6} | {'Conf':<6} | {'Sector':<16} | Signal")
    print("  " + "-" * 76)

    shorts = results_df.tail(top_n).iloc[::-1]
    for i, (_, row) in enumerate(shorts.iterrows(), 1):
        conf = "HIGH" if row['score'] < 0.35 else ("MED" if row['score'] < 0.45 else "LOW")
        print(f"  {i:>4} | {row['ticker']:<6} | {row['score']:.3f} | {conf:<6} | {row['sector']:<16} | {row['signal']}")

    print("\n" + "=" * 80)
    print("  Score: 0-1 scale | >0.65 = strong buy | <0.35 = strong short")
    print("=" * 80 + "\n")

    return buys, shorts

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fast Swing Stock Predictor")
    parser.add_argument('--train', action='store_true', help='Force retrain model')
    parser.add_argument('--top', type=int, default=5, help='Number of top picks (default 5)')
    parser.add_argument('--years', type=int, default=TRAIN_YEARS, help='Years of training data')
    args = parser.parse_args()

    # 1. Get tickers
    print("Fetching S&P 500 tickers...")
    tickers, sectors = get_sp500_tickers()
    print(f"Found {len(tickers)} tickers")

    # 2. Download data
    data = download_data(tickers, years=args.years)

    # 3. Build features
    combined, all_features, close_dict = build_dataset(data, tickers, sectors)

    # 4. Train or load model
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

    # 5. Generate today's picks
    print("\nGenerating predictions...")
    results = predict_today(model, all_features, sectors)

    # 6. Output
    buys, shorts = print_picks(results, top_n=args.top)

    # 7. Save to CSV
    results.to_csv("predictions.csv", index=False)
    print("Full rankings saved to predictions.csv")

if __name__ == "__main__":
    main()
