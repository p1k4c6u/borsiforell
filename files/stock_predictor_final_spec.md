# Fast Swing Stock Predictor — FINAL SPEC

## Goal
Predict S&P 500 stocks that will outperform/underperform over **3-5 trading days**. Output top 5 buys + top 5 shorts daily.

## Data
- **Source**: `yfinance` daily OHLCV
- **Universe**: S&P 500, volume > 1M/day
- **Training**: 2-3 years daily data
- **Skip**: Stocks with earnings in next 5 days

## Features (25)
- **Momentum (6)**: 1d/3d/5d/10d returns, 5d vs 20d acceleration, overnight gap
- **Mean Reversion (5)**: RSI(5), RSI(14), Bollinger Band position, dist from 5d/20d high
- **Volume (5)**: Vol ratio 5d/20d, vol spike, OBV slope, vol-price divergence, VWAP distance
- **Volatility (4)**: ATR%, intraday range, vol compression, close position in range
- **Sector (5)**: Sector 3d return, sector breadth, rank in sector, SPY 3d return, sector vs market

## Target
Forward 5-day return quintile (1-5). Buy Q4-Q5. Short Q1-Q2.

## Model
60% LightGBM (depth=5, regularized) + 40% momentum composite score.

## Output
Top 5 BUY + Top 5 SHORT with scores and signal reasons.

## Stack
`yfinance`, `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `ta`
