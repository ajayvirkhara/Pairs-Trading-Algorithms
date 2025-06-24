# 📘 Advanced Pairs Trading Algorithm

## 0. Features

- Cointegration-based statistical arbitrage on S&P 500 sector tickers
- Automatically selects top 10 cointegrated pairs using in-sample training
- Rolling regression (RollingOLS) to estimate hedge ratios
- Volatility-based position sizing
- Grid-search optimization of entry/exit thresholds (Sharpe-maximizing)
- Diversified capital allocation across multiple long/short pairs
- Optional transaction costs
- Full backtest evaluation with CAGR, Sharpe Ratio, and Max Drawdown

---

## 1. Overview

This refined pairs trading backtest extends the original MSc version by incorporating more robust elements for signal generation and capital allocation. It supports dynamic hedge ratio estimation, optimized entry/exit thresholds via grid-search, and capital sizing based on spread volatility. All trades are generated across top 10 cointegrated pairs per sector, evaluated over a test window.

---

## 2. Backtest Setup

- **Date range:** 2014-01-01 to 2021-12-01
- **Train/test split:** 75% train (~6 years), 25% test (~2 years)
- **Data source:** Yahoo Finance (monthly adjusted close)
- **Sector filter:** configurable via `SECTOR` env var or code

---

## 3. Configuration

Set these constants at the top of the script:

```python
TEST_SIZE          = 0.25               # Train/test split
START              = "2014-01-01"       # Start date
END                = "2021-12-01"       # End date
TOTAL_CAPITAL      = 1000               # Total backtest capital
IN_SAMPLE_YEARS    = 5                  # Min training overlap (in years)
ROLL_WINDOW_BETA   = 24                 # Rolling hedge regression window (months)
ROLL_WINDOW_VOL    = 12                 # Spread volatility sizing window (months)
DATA_FREQ          = "1mo"              # Price data interval
SECTOR             = "Real Estate"      # GICS sector filter
TRANSACTION_COST   = 0.000              # Transaction cost per trade

```
---

## 4. Algorithm Steps

### Step 1: Ticker Universe

- Scrape S&P 500 company list from Wikipedia
- Filter tickers by selected `SECTOR`

### Step 2: Price Fetching

- Download adjusted close prices via `yfinance`
- Use monthly data (`interval='1mo'`) and forward fill gaps
- Drop tickers with missing or zero data

### Step 3: Train/Test Split

- 75% of data used for cointegration testing & model fitting
- Remaining 25% used for trading/backtesting

### Step 4: Cointegration Analysis

- Loop over all ticker pairs
- Filter out assets with less than 60 months of overlap
- Apply Augmented Dickey-Fuller cointegration test
- Select top 10 pairs with p-value < 0.05
- If no pairs exist with p-value < 0.05, return "No cointegrated pairs found."

### Step 5: Rolling Regression

- For each pair, fit `RollingOLS` over training set to compute rolling hedge ratio (beta)
- Use final rolling beta as hedge ratio in test set

### Step 6: Spread & Volatility Sizing

- Compute spread = asset2 - beta \* asset1
- Calculate rolling z-score of spread
- Estimate rolling spread volatility to size positions (1% of capital per std dev)

### Step 7: Threshold Optimization

- Run grid search over entry/exit z-score thresholds
- For each combination, compute position signals and portfolio returns
- Select thresholds maximizing Sharpe ratio

### Step 8: Signal Generation

- Apply best entry/exit thresholds
- Generate long/short/flat signals
- Track changes in position (trades)

### Step 9: Trade Simulation

- Compute daily PnL from position changes and returns
- Incorporate transactions costs as defined
- Build equity curve for each pair

### Step 10: Portfolio Aggregation

- Combine equity curves from all 10 pairs
- Compute final portfolio value, CAGR, Sharpe Ratio, and Max Drawdown

---

## 5. Visualization

- Per-pair signal plot: entry/exit trades overlaid on price chart
- Combined portfolio chart: total equity curve and monthly return bars

---

## 6. Output Metrics

- Backtest start date
- Backtest duration (in days)
- CAGR
- Sharpe Ratio (monthly returns, annualized)
- Max Drawdown
- Final Portfolio Value

---

## 7. Requirements

Install dependencies:

```bash
pip install -r requirements_advanced.txt
```
---

## 8. Usage

```bash
cd advanced_algorithm
python pairs_trading_advanced.py
```

## 9. Author
Ajayvir Khara  
[LinkedIn](https://linkedin.com/in/ajayvirkhara)  
[GitHub](https://github.com/ajayvirkhara)

---

## 10. License
MIT © 2025

