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

## 📑 Table of Contents
0. [Features](#0-features)  
1. [Overview](#1-overview)  
2. [Key Results](#2-key-results)  
3. [Prerequisites](#3-prerequisites)  
4. [Repository Structure](#4-repository-structure)  
5. [Configuration](#5-configuration)  
6. [Usage](#6-usage)  
7. [License & Contact](#7-license--contact)  

## 1. Overview  
A repository of two pairs-trading backtests on S&P 500 sectors during the COVID-19 crisis. The original version was developed as part of my MSc dissertation, with the subsequent advanced algorithm incorporating refinements reflecting deeper exploration of strategy design and implementation. Both use cointegration to identify mean-reverting pairs; the advanced version adds rolling regressions, grid-search threshold optimization, volatility sizing, stop-loss/take-profit logic and top-10 pair diversification.

## 2. Key Results (2011-01-01 – 2021-01-01, 75/25 train/test, frictionless trading)  
| Sector                   | CAGR: MSc. Alg. (top pair)       | Sharpe | CAGR: Adv. Alg. (top 10 pairs)    | Sharpe |
|--------------------------|----------------------------------|--------|-----------------------------------|--------|
| Energy                   | CVX-OKE: 11.4 %    	          | 0.82   | 12.1 %           	               | 0.63   |
| Materials                | AMCR-MLM: 4.5 %        	      | 1.08   | 6.4 %                             | 1.36   |
| Industrials              | AOS-LII: 5.4 %         	      | 0.48   | 7.4 %                 		       | 1.85   |
| Consumer Discretionary   | DHI-EBAY: 7.6 %        	      | 1.02   | 4.6 %                		       | 0.89   |
| Consumer Staples         | DLTR-MO: 5.3 %         	      | 0.34   | 4.0 %                		       | 0.84   |
| Health Care              | LH-UNH: 2.9 %          	      | 0.32   | 2.2 %               		       | 0.54   |
| Financials               | ICE-WTW: 3.3 %         	      | 1.33   | 6.0 %                		       | 1.62   |
| Information Technology   | AAPL-VRSN: –3.7 %      	      | –0.79  | 7.2 %               		       | 2.00   |
| Comm. Services           | MTCH-TKO: –31.2 %      	      | –1.74  | 4.0 %                	 	       | 0.69   |
| Utilities                | AEP-ES: 1.4 %          	      | 0.64   | 3.5 %                 	           | 1.85   |
| Real Estate              | ESS-EXR: –0.73 %       	      | –0.13  | 6.5 %                	           | 1.53   |

## 3. Prerequisites  
- Python 3.8+  
- `pip install -r requirements_msc.txt`  
- `pip install -r requirements_advanced.txt`  

## 4. Repository Structure

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

## 5. Configuration  

Edit constants at top of each script:  
- `START`, `END` – Backtest window: default 2011-01-01, 2021-01-01  
- `TEST_SIZE` – Train/test split: default 0.25
- `SECTOR` – GICS sector filter 
- `DATA_FREQ` – Frequency of price data: default "1mo"

### ⚙️ MSc specific config
- `CUTOFF_DATE` – Minimum data cutoff: default "2018-01-01"
- `MIN_MONTHS` – Required monthly history: default 60
- `INITIAL_CAP_PER_ASSET` – Capital per trade leg: default 500

### 🔬 Advanced specific Config
- `ROLL_WINDOW_BETA` – Rolling regression window size: default 24
- `ROLL_WINDOW_VOL` – Volatility sizing lookback: default 12
- `TOTAL_CAPITAL` – Capital allocated across all pairs: default 10000
- `TRANSACTION_COST` – Transaction costs per trade: default 0

## 6. Usage  

cd Pairs-Trading-Algorithms/msc_algorithm
python pairs_trading_msc.py

cd  Pairs-Trading-Algorithms/advanced_algorithm
python pairs_trading_advanced.py

## 7. License & Contact
## 📜 License
MIT © 2025

## 🙋 Contact
**Ajayvir Khara**  
[LinkedIn](https://linkedin.com/in/ajayvirkhara)  
[GitHub](https://github.com/ajayvirkhara)