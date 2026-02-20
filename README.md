# Fast Swing Stock Predictor

Predicts S&P 500 stocks likely to outperform or underperform over **3-5 trading days**.
Outputs top BUY + SHORT picks daily using a LightGBM + momentum ensemble model.

## Live Dashboard

> **[View daily picks on GitHub Pages](https://p1k4c6u.github.io/borsiforell/)**

The dashboard is updated automatically every weekday after US market close via GitHub Actions.

## How It Works

```
S&P 500 OHLCV (yfinance)
        ↓
25 Technical Features
  Momentum · Mean Reversion · Volume · Volatility · Sector
        ↓
LightGBM Classifier  +  Momentum Composite Score
     (60%)                      (40%)
        ↓
Ensemble Score → Top 5 BUYs + Top 5 SHORTs
```

**Features (25):** 1d/3d/5d/10d returns, RSI(5/14), Bollinger Bands, ATR, OBV slope, VWAP distance, vol spike, sector breadth, rank-in-sector, and more.

**Target:** Forward 5-day return quintile (cross-sectional ranking per date). Buy Q4-Q5, short Q1-Q2.

## Local Usage

```bash
# Install dependencies
pip install -r files/requirements.txt

# Run with defaults (retrain if no model found)
python files/stock_predictor.py

# Force retrain
python files/stock_predictor.py --train

# Show top 10 picks + save JSON for the dashboard
python files/stock_predictor.py --top 10 --json

# Use 1 year of training data (faster)
python files/stock_predictor.py --years 1
```

## GitHub Pages Setup

1. Push this repo to GitHub
2. Go to **Settings → Pages → Source**: deploy from branch `main`, folder `/docs`
3. The GitHub Actions workflow (`.github/workflows/daily_predictions.yml`) runs every weekday at 9:30 PM UTC and commits fresh predictions to `docs/predictions.json`
4. The dashboard at `docs/index.html` reads that JSON and renders it

## Stack

`yfinance` · `pandas` · `numpy` · `lightgbm` · `scikit-learn` · `ta`

## Disclaimer

For educational purposes only. Not financial advice. All investing involves risk.
