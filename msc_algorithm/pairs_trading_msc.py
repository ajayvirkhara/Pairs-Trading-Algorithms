# ──────────────────────────────────────────────────────────────────────────────
# Imports.

import pandas as pd
import datetime
from datetime import timedelta
import os
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# Strategy set-up.

TEST_SIZE             = 2/8                                  # fraction of data held out for testing
MIN_MONTHS            = 60                                   # months of history required for cointegrated pairs
CUTOFF_DATE           = pd.Timestamp("2019-12-01")           # cut-off date for requiring MIN_MONTHS of history before testing cointegration
INITIAL_CAP_PER_ASSET = 500                                  # capital allocated per asset
START                 = datetime.datetime(2014, 1, 1)        # backtest window start date
END                   = datetime.datetime(2021, 12, 1)       # backtest window end date
DATA_FREQ             = os.getenv("DATA_FREQ", "1mo")        # price data frequency
SECTOR                = os.getenv("SECTOR", "Energy")        # GICS sector

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
if df_prices.empty:
    print("Error: No valid price data retrieved. Check tickers.")
else:
    print(df_prices.head())

# ──────────────────────────────────────────────────────────────────────────────
# Training / Testing split.

train_close, test_close = train_test_split(df_prices, test_size=TEST_SIZE, shuffle=False)

# ──────────────────────────────────────────────────────────────────────────────
# Exploratory plots: correlation heatmap (filtered to |ρ| > 0.6).

corr = train_close.pct_change(fill_method=None).corr(method='pearson')
high_corr = (
    corr
    .where(corr.abs() > 0.6)
    .dropna(axis=0, how='all')
    .dropna(axis=1, how='all')
)
fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(high_corr, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, ax=ax)
ax.set_title('Filtered Assets Correlation Matrix (|ρ|>0.6)')

# ──────────────────────────────────────────────────────────────────────────────
# Cointegration function.

def find_cointegrated_pairs(data):
    n = data.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = pd.DataFrame(columns=['asset1','asset2'])
    for i in range(n):
        for j in range(i+1, n):
            # Align the two series and drop any row with NA/Inf values in either.
                pair_df = (pd.concat(
                [ data[keys[i]].replace([np.inf, -np.inf], np.nan),
                data[keys[j]].replace([np.inf, -np.inf], np.nan) ],axis=1).dropna())

            # Only consider pairs with atleast 60 months of price history before 2018-01-01.
                cutoff = CUTOFF_DATE
                valid = pair_df.loc[pair_df.index < cutoff].dropna().shape[0]
                if valid <= MIN_MONTHS:
                    continue

            # Skip the pair if there's too little data to test or no variation in data.
                if pair_df.shape[0] < 3 or pair_df.iloc[:,0].nunique() < 2 or pair_df.iloc[:,1].nunique() < 2:
                    continue

             # Perform cointegration test, storing ticker-pairs where p-value < 0.05.
                result = coint(pair_df.iloc[:,0], pair_df.iloc[:,1])
                pvalue_matrix[i, j] = result[1]
                if result[1] < 0.05:
                    new_pair = pd.DataFrame({'asset1': [keys[i]],'asset2': [keys[j]]})
                    pairs = pd.concat([pairs, new_pair], ignore_index=True)
    return pvalue_matrix, pairs


# ──────────────────────────────────────────────────────────────────────────────
# Cointegration results.

pvalues, pairs = find_cointegrated_pairs(train_close)
pvalue_df = pd.DataFrame(
    pvalues,
    index=train_close.columns,
    columns=train_close.columns
)

# Rank asset-pairs in ascending p-value order.
best = (pairs
        .assign(pvalue=[pvalue_df.loc[a,b] for a,b in zip(pairs.asset1, pairs.asset2)])
        .sort_values('pvalue'))
print("Top 10 cointegrated pairs:\n", best.head(10))

# Plot p-values as a heatmap.
fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(pvalues, xticklabels = train_close.columns,
                yticklabels = train_close.columns, cmap = 'RdYlGn_r', annot = True, fmt=".2f",
                mask = (pvalues >= 0.05))
ax.set_title('Cointregration Matrix p-values Between Pairs')
plt.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# Backtest set-up.

# Choose asset-pair.
asset1 = 'PSX'
asset2 = 'TPL'

# Build training dataframe for the chosen pair and drop any NaN/inf values.
train = pd.DataFrame()
train['asset1'] = train_close[asset1]
train['asset2'] = train_close[asset2]
train = train.replace([np.inf, -np.inf], np.nan).dropna()

# Run OLS regression and print results.
model=sm.OLS(train.asset2, train.asset1).fit()
plt.rc('figure', figsize=(12, 7))
fig, ax = plt.subplots(figsize=(12, 7))
ax.axis("off") 
summary_text = model.summary().as_text()
plt.text(0.01, 0.99, summary_text, fontsize=12, fontproperties="monospace", verticalalignment="top")
plt.tight_layout()
print('Hedge Ratio = ', model.params.iloc[0])

# Calculate and plot spread.
spread = train.asset2 - model.params.iloc[0] * train.asset1
plt.figure(figsize=(12,6))
ax = spread.plot(title = "Pair's Spread")
ax.set_ylabel("Spread")
ax.grid(True);
ax.margins(y=0.1)
plt.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# Signal generation & PnL.

# Calculate z-score.
def zscore(series):
    return(series - series.mean()) / np.std(series)

# Create a dataframe for trading signals.
signals = pd.DataFrame()
signals['asset1'] = test_close[asset1] 
signals['asset2'] = test_close[asset2]
ratios = signals.asset1 / signals.asset2

# Calculate z-score and define upper and lower thresholds.
signals['z'] = zscore(ratios)
signals['z upper limit'] = np.mean(signals['z']) + 1*np.std(signals['z'])
signals['z lower limit'] = np.mean(signals['z']) - 1*np.std(signals['z'])

# Create signal - short if z-score is greater than upper limit else long.
signals['signals1'] = 0
signals['signals1'] = np.select([signals['z'] > \
                                signals['z upper limit'], signals['z'] < signals['z lower limit']], [-1, 1], default=0)

# Generate trade actions by differencing the target position signals for each asset.
signals['positions1'] = signals['signals1'].diff()
signals['signals2'] = -signals['signals1']
signals['positions2'] = signals['signals2'].diff()
pd.concat([signals.head(3),signals.tail(3)])

# Visualize trading signals and position.
fig=plt.figure(figsize=(14,6))
bx = fig.add_subplot(111)   
bx2 = bx.twinx()

# Plot the two different assets.
l1, = bx.plot(signals['asset1'], color='darkblue')
l2, = bx2.plot(signals['asset2'], color='orangered')
u1, = bx.plot(signals['asset1'][signals['positions1'] == 1], lw=0, marker='^', markersize=8, c='g',alpha=0.7)
d1, = bx.plot(signals['asset1'][signals['positions1'] == -1], lw=0,marker='v',markersize=8, c='r',alpha=0.7)
u2, = bx2.plot(signals['asset2'][signals['positions2'] == 1], lw=0,marker=2,markersize=9, c='g',alpha=0.9, markeredgewidth=3)
d2, = bx2.plot(signals['asset2'][signals['positions2'] == -1], lw=0,marker=3,markersize=9, c='r',alpha=0.9,markeredgewidth=3)
bx.set_ylabel(asset1,)
bx2.set_ylabel(asset2, rotation=270)
bx.yaxis.labelpad=15
bx2.yaxis.labelpad=15
bx.set_xlabel('Date')
bx.xaxis.labelpad=15
plt.legend([l1,l2,u1,d1,u2,d2], [asset1, asset2,'LONG {}'.format(asset1),
           'SHORT {}'.format(asset1),
           'LONG {}'.format(asset2),
           'SHORT {}'.format(asset2)], loc ='best')
plt.title('Opening/Closing of Trading Positions with |z-score|> 1')
plt.xlabel('Date')
plt.grid(True)
plt.tight_layout()

# Shares to buy for each position.
positions1 = INITIAL_CAP_PER_ASSET// max(signals['asset1'])
positions2 = INITIAL_CAP_PER_ASSET// max(signals['asset2'])

# Since there are two assets, we calculate each asset PnL separately and in the end we aggregate them into one portfolio.
# PnL for the 1st asset.
portfolio = pd.DataFrame()
portfolio['asset1'] = signals['asset1']
portfolio['holdings1'] = signals['positions1'].cumsum() * signals['asset1'] * positions1
portfolio['cash1'] = INITIAL_CAP_PER_ASSET - (signals['positions1'] * signals['asset1'] * positions1).cumsum()
portfolio['total asset1'] = portfolio['holdings1'] + portfolio['cash1']
portfolio['return1'] = portfolio['total asset1'].pct_change()
portfolio['positions1'] = signals['positions1']

# PnL for the 2nd asset.
portfolio['asset2'] = signals['asset2']
portfolio['holdings2'] = signals['positions2'].cumsum() * signals['asset2'] * positions2
portfolio['cash2'] = INITIAL_CAP_PER_ASSET - (signals['positions2'] * signals['asset2'] * positions2).cumsum()
portfolio['total asset2'] = portfolio['holdings2'] + portfolio['cash2']
portfolio['return2'] = portfolio['total asset2'].pct_change()
portfolio['positions2'] = signals['positions2']

# Total PnL and z-score.
portfolio['z'] = signals['z']
portfolio['total asset'] = portfolio['total asset1'] + portfolio['total asset2']
portfolio['z upper limit'] = signals['z upper limit']
portfolio['z lower limit'] = signals['z lower limit']
portfolio = portfolio.dropna()

# Plot the change in portfolio value, PnL, and z-score.
fig = plt.figure(figsize=(14,6),)
ax = fig.add_subplot(111)
ax2 = ax.twinx()
l1, = ax.plot(portfolio['total asset'], c='g')
l2, = ax2.plot(portfolio['z'], c='black', alpha=0.3)
b = ax2.fill_between(portfolio.index,portfolio['z upper limit'],\
                portfolio['z lower limit'], \
                alpha=0.2,color='#ffb48f')
ax.set_ylabel('Portfolio Value')
ax2.set_ylabel('Z Statistics',rotation=270)
ax.yaxis.labelpad=15
ax2.yaxis.labelpad=15
ax.set_xlabel('Date')
ax.xaxis.labelpad=15
plt.title('Portfolio Performance with Profit and Loss')
plt.legend([l2,b,l1],['Z Statistics', 'Z Statistics +-1 Sigma', 'Total Portfolio Value'],loc='upper left');

# ──────────────────────────────────────────────────────────────────────────────
# Results output.
final_portfolio = portfolio['total asset'].iloc[-1]
delta = (portfolio.index[-1] - portfolio.index[0]).days
print('Number of days : ', delta)
backtest_start = test_close.index[0].date()
print('Backtest start date : ', backtest_start)
YEAR_DAYS = 365
returns = (final_portfolio/1000) ** (YEAR_DAYS/delta) - 1
print('CAGR = {:.2f}%' .format(returns * 100))
monthly_rets = portfolio['total asset'].pct_change().dropna()
sharpe = monthly_rets.mean() / monthly_rets.std() * np.sqrt(12)
print(f"Sharpe Ratio: {sharpe:.2f}")

# Final portfolio value.
print('Final Portfolio Value = ${:.3f}' .format(final_portfolio))