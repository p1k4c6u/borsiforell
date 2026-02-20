# Fast Swing Stock Predictor

## Quick Start
```bash
pip install -r requirements.txt
python stock_predictor.py --train --top 5
```

## What It Does
Scans all S&P 500 stocks, computes 25 features (momentum, volume, mean reversion, volatility, sector signals), runs a LightGBM model ensembled with a momentum baseline, and outputs:
- **Top 5 BUY picks** (predicted top quintile over next 3-5 days)
- **Top 5 SHORT picks** (predicted bottom quintile)

## Commands
```bash
python stock_predictor.py --train        # First run: download data + train model
python stock_predictor.py                # Re-run with saved model
python stock_predictor.py --top 10       # Show top 10 buys + 10 shorts
python stock_predictor.py --years 3      # Use 3 years of training data
```

## Files Created
- `model_lgbm.pkl` — Saved trained model
- `predictions.csv` — Full ranked list of all stocks

## Notes
- First run takes 5-10 min (downloading 500 stocks)
- Re-runs with saved model are fast (~1 min)
- Retrain weekly for best results
