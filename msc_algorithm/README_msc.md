# 📘 MSc Pairs Trading Algorithm

## 1. Description
This script implements the original version of a pairs trading strategy developed as part of my MSc dissertation. It tests for cointegration between S&P 500 stocks within a specified sector and simulates a simple mean-reversion trading strategy based on z-score thresholds.

---

## 2. Methodology Overview
- **Universe**: S&P 500 stocks from a selected GICS sector.
- **Data Source**: Yahoo Finance (via `yfinance`), monthly adjusted close prices.
- **Cointegration Filter**: Pairs with ≥ 60 months of overlapping data pre-2019-12-01 and p-value < 0.05.
- **Regression**: Static OLS on training period.
- **Signals**: Z-score of price ratio. Long if z < -1, short if z > +1.
- **Capital Allocation**: $500 per asset.

---

## 3. Key Parameters
| Variable               | Description                                      | Default                  |
|------------------------|--------------------------------------------------|--------------------------|
| `START`, `END`         | Backtest window                                  | 2014-01-01 to 2021-12-01 |
| `TEST_SIZE`            | Fraction of data held out for testing            | 0.25                     |
| `SECTOR`               | GICS sector (env override supported)             | "Real Estate"            |
| `DATA_FREQ`            | Price data interval (env override supported)     | "1mo"                    |
| `CUTOFF_DATE`          | Cointegration history cutoff                     | 2019-12-01               |
| `MIN_MONTHS`           | Minimum history before cutoff                    | 60 months                |
| `INITIAL_CAP_PER_ASSET`| Capital per asset                                | 500                      |

---

## 4. Strategy Steps
### Step 1: Data Preparation
- Scrape S&P 500 tickers from Wikipedia.
- Filter tickers by GICS sector.
- Download monthly price data from Yahoo Finance.
- Drop tickers with missing data.

### Step 2: Train/Test Split
- Non-shuffled split (25% reserved as test period).

### Step 3: Correlation Pre-screen
- Pearson correlation heatmap filtered to |ρ| > 0.6 to visualize co-movement.

### Step 4: Cointegration Check
- For each ticker pair, compute Engle-Granger p-value.
- Retain pairs with sufficient history and p < 0.05.

### Step 5: Signal Generation
- Select a top cointegrated pair.
- Estimate hedge ratio via OLS.
- Compute spread and z-score.
- Generate signals:
  - **Long** if z < –1
  - **Short** if z > +1
  - **Exit** when z returns to within ±1

### Step 6: Trade and Portfolio Simulation
- Allocate fixed notional capital per asset.
- Track holdings, cash, and total asset value.
- Compute total portfolio PnL and Sharpe Ratio.

---

## 5. Outputs
- **Top 10 Cointegrated Pairs** ranked by p-value
- **Correlation Heatmap** and **Cointegration Matrix**
- **Spread Plot** for selected pair
- **Signal Plot** (entry/exit points)
- **Portfolio Value Chart** with z-score overlay
- **Output Metrics**:
  - Backtest start date
  - Backtest duration (in days)
  - CAGR
  - Sharpe Ratio
  - Final Portfolio Value

---

## 6. Usage
```bash
cd msc_algorithm
python pairs_trading_msc.py
```
Optional: override defaults
```bash
$env:SECTOR = "Financials"
$env:DATA_FREQ = "1mo"
python pairs_trading_msc.py
```

---

## 7. Limitations
- No stop-loss/take-profit logic
- Only backtests a **single** cointegrated pair
- No transaction costs or slippage
- Static hedge ratio
- Fixed z-score entry/exit thresholds

---

## 8. Author
Ajayvir Khara  
[LinkedIn](https://linkedin.com/in/ajayvirkhara)  
[GitHub](https://github.com/ajayvirkhara)

---

## 9. License
MIT © 2025

---

## ⚠️ Disclaimer

This repository is for educational and research purposes only. The algorithms and results are not intended as financial advice or investment recommendations. Past performance, especially in backtests, is not indicative of future results. Use of this code is at your own risk.
