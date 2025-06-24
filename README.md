# 📈 Pairs Trading Backtest

## 0. Features
- Cointegration-based strategy on S&P 500 sectors
- Original MSc version and postgrad-refined algorithm
- Rolling regression hedge ratios
- Grid-search entry/exit thresholds
- Volatility sizing and risk controls (stop-loss/take-profit)
- Sector filtering and top-10 pair diversification
- Performance comparison during COVID-era volatility
- Optional transaction costs

---

## 📑 Table of Contents
0. [Features](#0-features)  
1. [Overview](#1-overview)  
2. [Key Results](#2-key-results)  
3. [Prerequisites](#3-prerequisites)  
4. [Repository Structure](#4-repository-structure)  
5. [Configuration](#5-configuration)  
6. [Usage](#6-usage)  
7. [Author](#7-author)  
8. [License](#8-license)  

---

## 1. Overview  
A repository of two pairs-trading backtests on S&P 500 sectors during the COVID-19 crisis. The original version was developed as part of my MSc dissertation, with the subsequent advanced algorithm incorporating refinements reflecting deeper exploration of strategy design and implementation. Both use cointegration to identify mean-reverting pairs; the advanced version adds rolling regressions, grid-search threshold optimization, volatility sizing, stop-loss/take-profit logic and top-10 pair diversification.

---

## 2. Key Results (2014-01-01 – 2021-12-01, 75/25 train/test, frictionless trading)  
| Sector                   | CAGR: MSc. Alg. (top pair)       | Sharpe | CAGR: Adv. Alg. (top 10 pairs)    | Sharpe |
|--------------------------|----------------------------------|--------|-----------------------------------|--------|
| Energy                   | PSX-TPL: -9.6 %    	          | -0.30  | 10.6%   	                       |  0.76  |
| Materials                | AMCR-NEM: 13.9 %        	      |  0.84  | 10.1%                             |  1.78  |
| Industrials              | BR-EXPD: 4.5 %         	      |  1.75  | 7.5%                 		       |  1.87  |
| Consumer Discretionary   | AMZN-BBY: 4.5 %        	      |  0.47  | 8.8%               		       |  1.04  |
| Consumer Staples         | KDP-MKC: 3.4 %         	      |  0.56  | 8.9%              		           |  2.83  |
| Health Care              | ELV-RVTY: 7.0 %          	      |  0.52  | 11.3%             		           |  2.98  |
| Financials               | MTB-RJF: -8.9 %         	      | -0.89  | 8.7%              		           |  1.80  |
| Information Technology   | CSCO-LRCX: –3.9 %      	      | –0.24  | 12.5%          		           |  2.41  |
| Comm. Services           | CHTR-CMCSA: 4.8 %      	      |  0.50  | 12.8%           	 	           |  2.55  |
| Utilities                | ETR-SRE: 5.9 %          	      |  1.51  | 6.2%            	               |  2.31  |
| Real Estate              | CCI-CPT: –3.8 %        	      | –0.69  | 6.3%               	           |  1.70  |

---

## 3. Prerequisites  
- Python 3.8+  
- `pip install -r requirements_msc.txt`  
- `pip install -r requirements_advanced.txt`  

---

## 4. Repository Structure

```

Pairs-Trading-Algorithms/
├── msc_algorithm/
│ ├── pairs_trading_msc.py
│ ├── README_msc.txt
│ ├── requirements_msc.txt
├── advanced_algorithm/
│ ├── pairs_trading_advanced.py
│ ├── README_advanced.md
│ ├── requirements_advanced.txt
├── LICENSE
└── README.md

```

---

## 5. Configuration  

Edit constants at top of each script:  
- `START`, `END` – Backtest window: default 2014-01-01, 2021-12-01  
- `TEST_SIZE` – Train/test split: default 0.25 (roughly 2 years)
- `SECTOR` – GICS sector filter 
- `DATA_FREQ` – Frequency of price data: default "1mo"

### ⚙️ MSc specific config
- `CUTOFF_DATE` – Minimum data cutoff: default "2019-12-01"
- `MIN_MONTHS` – Required monthly history: default 60
- `INITIAL_CAP_PER_ASSET` – Capital per trade leg: default 500

### 🔬 Advanced specific Config
- `ROLL_WINDOW_BETA` – Rolling regression window size: default 24
- `ROLL_WINDOW_VOL` – Volatility sizing lookback: default 12
- `TOTAL_CAPITAL` – Capital allocated across all pairs: default 10000
- `TRANSACTION_COST` – Transaction costs per trade: default 0

---

## 6. Usage  

cd Pairs-Trading-Algorithms/msc_algorithm
python pairs_trading_msc.py

cd  Pairs-Trading-Algorithms/advanced_algorithm
python pairs_trading_advanced.py

---

## 7. Author
Ajayvir Khara  
[LinkedIn](https://linkedin.com/in/ajayvirkhara)  
[GitHub](https://github.com/ajayvirkhara)

---

## 8. License
MIT © 2025

---

## ⚠️ Disclaimer

This repository is for educational and research purposes only. The algorithms and results are not intended as financial advice or investment recommendations. Past performance, especially in backtests, is not indicative of future results. Use of this code is at your own risk.