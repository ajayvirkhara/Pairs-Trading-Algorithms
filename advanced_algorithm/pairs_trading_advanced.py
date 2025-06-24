# ──────────────────────────────────────────────────────────────────────────────
# Imports.

# Standard library.
import datetime
from datetime import timedelta
import os

# Third-party libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import coint
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

# ────────────────────────────────────────────────────────────────────────────── 
# Strategy set-up.

TEST_SIZE          = 2/8                                  # fraction of data held out for testing
START              = "2014-01-01"                         # backtest window start date
END                = "2021-12-01"                         # backtest window end date
TOTAL_CAPITAL      = 1000                                 # total capital allocated
IN_SAMPLE_YEARS    = 5                                    # years of history required for cointegrated pairs
ROLL_WINDOW_BETA   = 24                                   # months for rolling Beta estimation
ROLL_WINDOW_VOL    = 12                                   # months to estimate spread volatility
DATA_FREQ          = os.getenv("DATA_FREQ", "1mo")        # price data frequency
SECTOR             = os.getenv("SECTOR", "Energy")        # GICS sector
TRANSACTION_COST   = 0.000                                # transaction cost per trade

# ──────────────────────────────────────────────────────────────────────────────
# Scrape S&P 500 data from Wikipedia and rename columns.

wiki_url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_tbl = pd.read_html(wiki_url, header=0)[0]
sp500_tbl = sp500_tbl.rename(columns={
    "Symbol":       "ticker",
    "GICS Sector":  "sector",
    "GICS Sub-Industry": "industry"
})

# Verify table column names.
print(sp500_tbl.columns)  

# Extract relevant tickers for chosen sector and verify data.
sector    = SECTOR
sp_tickers = sp500_tbl.loc[
    sp500_tbl["sector"] == sector,
    "ticker"
].tolist()
print(f'Tickers in sector "{sector}": {', '.join(sp_tickers)}')

# ──────────────────────────────────────────────────────────────────────────────
# Fetching and cleaning auto-adjusted closing prices for the given data frequency using Yahoo Finance.

df_prices = yf.download(
        tickers=sp_tickers,
        start = START,
        end   = END,
        interval     = DATA_FREQ,
        auto_adjust  = False,
        progress     = False)
df_prices = (df_prices["Adj Close"]
             .dropna(axis=1, how="all") # Drop any tickers that never had data.
             .ffill()) # Use past value to fill any gaps in price history.
valid_tickers = df_prices.columns.tolist()
if df_prices.empty:
    print("Error: No valid price data retrieved. Check tickers.")
elif len(valid_tickers) < 10:
    print(f"Too few tickers with data: {valid_tickers}")
else:
    print(df_prices.head())
    
# ──────────────────────────────────────────────────────────────────────────────
# Training / Testing split.

train_close, test_close = train_test_split(df_prices, test_size=TEST_SIZE, shuffle=False)

# ──────────────────────────────────────────────────────────────────────────────
# Cointegration logic: loop over top 10 cointegrated pairs, build signals, optimize thresholds, and append equity curve.

def find_cointegrated_pairs(data):
    # Drop columns that are constant/near-constant.
    data = data.loc[:, data.nunique() > 2]

    keys = data.columns
    n = len(keys)
    pvals = np.ones((n, n))
    pairs = []

    for i in range(n):
        for j in range(i+1, n):
            # Align the two series & drop any NaN/Inf rows.
            pair_df = pd.concat([
                data[keys[i]].replace([np.inf, -np.inf], np.nan),
                data[keys[j]].replace([np.inf, -np.inf], np.nan)
            ], axis=1).dropna()

            # Require at least IN_SAMPLE_YEARS*12 months overlap (60 months by default).
            if pair_df.shape[0] < IN_SAMPLE_YEARS * 12:
                continue

            # Compute cointegration p‐value on cleaned data.
            p = coint(pair_df.iloc[:,0], pair_df.iloc[:,1])[1]
            pvals[i, j] = p

            # Store ticker-pairs where p-value < 0.05.
            if p < 0.05:
                pairs.append((keys[i], keys[j], p))

    # Return full matrix plus the top 10 sorted by p-value.
    pairs = sorted(pairs, key=lambda x: x[2])[:10]
    return pvals, pairs

_, top_pairs = find_cointegrated_pairs(train_close)
if not top_pairs:
    raise RuntimeError("No cointegrated pairs found.")

# Capital per pair.
capital_per_pair = TOTAL_CAPITAL / len(top_pairs)

# ──────────────────────────────────────────────────────────────────────────────
# Backtesting.

equity_curves = []
for asset1, asset2, p in top_pairs:
    initial_capital = capital_per_pair

    # Build signals DataFrame.
    signals = pd.DataFrame(index=test_close.index)
    signals['asset1'] = test_close[asset1]
    signals['asset2'] = test_close[asset2]

    # Rolling hedge ratio.
    train = train_close[[asset1, asset2]]
    roll = RollingOLS(train[asset2], train[asset1], window=ROLL_WINDOW_BETA).fit()
    # Use the last in‐sample β as the constant hedge ratio.
    hedge = roll.params[asset1].iloc[-1]
    signals['beta'] = hedge

    # Print the chosen hedge ratio
    print(f"Hedge ratio (beta) from RollingOLS: {hedge:.4f}")

    # (Optional) Refit a standard OLS on the last window to get the full regression summary
    last_window = train.tail(ROLL_WINDOW_BETA)
    ols_last = sm.OLS(last_window[asset2], last_window[asset1]).fit()
    print(ols_last.summary())
    
    # Define spread & z-score.
    spread = signals['asset2'] - signals['beta']*signals['asset1']
    signals['z'] = (spread - spread.mean())/spread.std()

    # Volatility sizing.
    signals['vol'] = spread.rolling(ROLL_WINDOW_VOL).std().bfill()
    signals['size1'] = (initial_capital*0.01) / signals['vol']
    signals['size2'] = signals['size1']

    # Cache returns once for efficiency.
    r1 = signals['asset1'].pct_change().fillna(0)
    r2 = signals['asset2'].pct_change().fillna(0)

    # Create array of entry/exit thresholds.
    enters = np.linspace(0.1, 2.5, 25)
    exits  = np.linspace(0.1, 2.5, 25)

    # Initialize best with default enter/exit.
    best = {
        'sharpe': -np.inf,
        'enter':  enters[0],
        'exit':   exits[0],
    }
    # Function for optimal grid-search entry/exit sigma threshold to maximize Sharpe.
    for enter in enters:
        for exit in exits:
            tmp_sig = signals[['z']].copy()
            tmp_sig['position1'] = np.where(tmp_sig.z> enter, -1,           # spread high -> short spread
                              np.where(tmp_sig.z< -enter, 1,                # spread low -> long spread
                              np.where(tmp_sig.z.abs()< exit, 0, np.nan)))  # close position when spread is near mean
            # Carry last valid position forward for any NaN values.
            tmp_sig['position1'] = tmp_sig['position1'].ffill().fillna(0)
            tmp_sig['position2'] = -tmp_sig['position1']

            tmp_sig['trade1'] = tmp_sig['position1'].diff().fillna(0)
            tmp_sig['trade2'] = tmp_sig['position2'].diff().fillna(0)

            # Simulate PnL via returns.
            eq = initial_capital * (
                1 + (tmp_sig['trade1'].shift(fill_value=0)*r1).cumsum()
                  + (tmp_sig['trade2'].shift(fill_value=0)*r2).cumsum()
            )
            # Compute monthly Sharpe and annualise.
            dr = eq.pct_change().dropna()
            std_dr = dr.std()
            if std_dr > 0:
                sharpe = dr.mean()/std_dr*np.sqrt(12)
                # Keep best trading thresholds.
                if sharpe>best['sharpe']:
                    best.update({'sharpe':sharpe,'enter':enter,'exit':exit})

    # Apply best thresholds.
    e, x = best['enter'], best['exit']

    # Print strategy pair and entry/exit thresholds.
    print(f"Backtesting pair: {asset1}-{asset2}")
    print(f"  Entry threshold: +/-{e:.2f} sigma, Exit threshold: {x:.2f} sigma")

    # Rebuild spread and Z with chosen Beta/Prices.
    spread = signals['asset2'] - signals['beta'] * signals['asset1']

    # Position 1 definition: +1 long asset2/short asset1; inverse for position 2.
    signals['position1'] = np.where(signals.z > e, -1,
                  np.where(signals.z < -e, 1, 0))
    signals['position1'] = signals['position1'].ffill().fillna(0)
    # Lock in position for atleast one time period.
    signals['position1'] = signals['position1'].rolling(window=2, min_periods=1).apply(lambda x: x.iloc[-1])    
    signals['position2'] = -signals['position1']
    signals['trade1'] = signals['position1'].diff().fillna(0)
    signals['trade2'] = signals['position2'].diff().fillna(0)
    
    # ─────────────────────────────────────────────────────────
    # Visualize trading signals and positions for this pair.
    fig, bx = plt.subplots(figsize=(14,6))
    bx2 = bx.twinx()

    # Plot price series.
    l1, = bx.plot(signals['asset1'], label=asset1, color='darkblue')
    l2, = bx2.plot(signals['asset2'], label=asset2, color='orangered', alpha=0.8)

    # Mark entry/exit on asset1.
    entries = signals.index[signals['trade1'] == 1]
    exits   = signals.index[signals['trade1'] == -1]
    entry_marker = bx.scatter(entries, signals.loc[entries, 'asset1'], marker='^', color='g', label='Entry')
    exit_marker  = bx.scatter(exits,   signals.loc[exits,   'asset1'], marker='v', color='r', label='Exit')

    bx.set_title(f"{asset1}-{asset2} Signals")
    bx.set_xlabel("Date")
    bx.set_ylabel(asset1)
    bx2.set_ylabel(asset2)
    bx.legend(loc='upper left', handles=[l1, entry_marker, exit_marker])
    plt.tight_layout()

    print(f"  z-range: {signals.z.min():.2f} to {signals.z.max():.2f}")
    print("  position1 unique:", np.unique(signals['position1'].fillna(0)))

    # Build final equity.
    r1 = signals['asset1'].pct_change().fillna(0)
    r2 = signals['asset2'].pct_change().fillna(0)
    transaction_cost = TRANSACTION_COST
    ret1 = signals['trade1'].shift(fill_value=0) * r1 - (transaction_cost * signals['trade1'].abs())
    ret2 = signals['trade2'].shift(fill_value=0) * r2 - (transaction_cost * signals['trade1'].abs())
    equity = initial_capital * (1 + (ret1 + ret2)).cumprod()
    equity_curves.append(equity)

# Combine and evaluate.
combined = sum(equity_curves)
days = (combined.index[-1] - combined.index[0]).days
cagr = (combined.iloc[-1]/TOTAL_CAPITAL)**(365/days) - 1
dr = combined.pct_change().dropna()
std_dr = dr.std()
sharpe = dr.mean()/std_dr*np.sqrt(12) if std_dr>0 else np.nan # avoids division by 0
cum = (1+dr).cumprod()
max_dd = ((cum - cum.cummax())/cum.cummax()).min()
END = combined.index[-1]
test_start = END - timedelta(days=days)

# ──────────────────────────────────────────────────────────────────────────────
# Visualize combined portfolio value and monthly returns.

# Compute monthly returns for plotting.
returns = combined.pct_change().fillna(0)

fig, ax = plt.subplots(figsize=(14,6))
ax2 = ax.twinx()

# Equity curve.
l1, = ax.plot(combined.index, combined, label='Portfolio Value', color='g')

# Bar‐style monthly returns.
# First compute dynamic bar width to ensure correct scaling.
period_days = (combined.index[1] - combined.index[0]).days
bar_width = period_days * 0.8

l2 = ax2.bar(
    combined.index,
    returns,
    width=bar_width,
    alpha=0.3,
    label='Monthly Return'
)

ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value')
ax2.set_ylabel('Monthly Return')

ax.set_title('Combined Portfolio Value & Monthly Returns')

# Legends.
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# Results output.

print(f"Backtest start date: {test_start.date()}")
print(f"Number of days : {days:.0f}")
print(f"Combined final value: ${combined.iloc[-1]:.2f}")
print(f"CAGR: {cagr:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")