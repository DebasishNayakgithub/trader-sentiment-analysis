# Primetrade.ai — Trader Performance vs Market Sentiment
**Data Science Intern Assignment — Round 0 | REAL DATA SUBMISSION**

> **Datasets:** Bitcoin Fear/Greed Index (`fear_greed_index.csv`) × Hyperliquid Historical Trader Data (`historical_data_csv.gz`)
> **Window:** May 2023 – May 2025 | 731 days | 211,224 trades | 32 accounts | 246 coins

---

## Quick Start

```bash
# 1. Place both CSVs in the same directory as the script
#    fear_greed_index.csv
#    historical_data_csv.gz

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter

# 3a. Run standalone analysis script (generates all charts + prints all stats)
python real_analysis.py

# 3b. OR open the interactive notebook
jupyter notebook trader_sentiment_analysis_REAL.ipynb
```

> **Data paths in code:** Update the two `pd.read_csv(...)` paths at the top of either file to point to your local copies of the datasets.

---

## Project Structure

```
primetrade_project/
├── trader_sentiment_analysis_REAL.ipynb   # Main notebook — all parts A/B/C + bonus
├── real_analysis.py                       # Standalone Python script (identical analysis)
├── README.md                              # This file
├── data/
│   ├── fear_greed_index.csv               # Bitcoin Fear/Greed Index (2,644 rows)
│   └── historical_data_csv.gz             # Hyperliquid trader data (211,224 rows)
└── charts/
    ├── chart1_pnl_winrate.png             # PnL histogram + 5-class win rate
    ├── chart2_behavior_shifts.png         # Behavioral metric bars by sentiment
    ├── chart3_account_segments.png        # Account scatter + cross-margin + drawdown
    ├── chart4_timeseries.png              # FG Index vs market PnL time-series
    ├── chart5_long_ratio_coins.png        # Long-ratio violin + top coin PnL
    ├── chart6_clusters.png               # KMeans behavioral archetypes
    ├── chart7_per_account_sentiment.png   # Per-account heatmap + ΔPnL chart
    └── chart8_feature_importance.png      # RF model feature importance
```

---

## Part A — Data Preparation

### Dataset Inventory

| Dataset | Rows | Columns | Period |
|---------|------|---------|--------|
| Fear/Greed Index | 2,644 | 4 (timestamp, value, classification, date) | Feb 2018 – May 2025 |
| Hyperliquid Trades | 211,224 | 16 (Account, Coin, Price, Size, Side, Timestamp IST, Direction, ClosedPnL, Fee, …) | May 2023 – May 2025 |

**Missing values:** Zero in both datasets.  
**Duplicates:** Zero in both datasets.  
**Alignment window:** May 2023 – May 2025 (731 overlapping days). 6 trader rows had no FG match (< 0.003% — dropped).

### Key Derived Features

| Feature | Definition |
|---------|-----------|
| `daily_pnl` | Sum of `Closed PnL − Fee` per account per day |
| `win_rate` | Fraction of closing trades with `Closed PnL > 0` |
| `n_trades` | Total fills per account per day |
| `long_ratio` | Fraction of BUY-side fills |
| `cross_margin_pct` | Fraction of trades with `Crossed == True` |
| `max_drawdown` | Min(cumulative PnL − running peak) per account lifetime |
| `total_vol_usd` | Sum of `Size USD` per account per day |

Sentiment binary: `Extreme Fear / Fear → "Fear"`, `Extreme Greed / Greed → "Greed"`, `Neutral → "Neutral"`

---

## Part B — Analysis

### B1. Performance: Fear vs Greed Days

| Sentiment | Trader-Days | Avg Daily PnL | Median Daily PnL | Win Rate | Avg Trades/Day | Avg Volume/Day |
|-----------|-------------|---------------|------------------|----------|----------------|----------------|
| **Fear**  | 790 | **$5,037** | $104 | 84.2% | **105** | **$756,720** |
| Neutral | 376 | $3,334 | $151 | 83.5% | 100 | $479,367 |
| Greed | 1,174 | $4,067 | **$236** | **85.6%** | 77 | $351,829 |

**Mann-Whitney U-test (Fear vs Greed daily PnL): p = 0.0162 → statistically significant.**

**Counterintuitive finding:** Mean daily PnL is *highest on Fear days* ($5,037 vs $4,067 on Greed). However, the *median* is much lower ($104 vs $236) — Fear days have a highly skewed distribution driven by a few very large wins. Greed days produce more consistent, smaller positive returns (as shown in the histogram).

**5-class breakdown:** Extreme Fear and Extreme Greed both show elevated PnL — traders are most active and capture larger moves during high-conviction sentiment regimes. Neutral is the weakest environment.

### B2. Behavioral Shifts by Sentiment

| Sentiment | Avg Trades/Day | Avg Volume | Long Ratio | Cross-Margin % |
|-----------|----------------|------------|------------|----------------|
| Fear | 105 (+37% vs Greed) | $756K (+115% vs Greed) | **52.2%** (slight long bias) | 65.1% |
| Neutral | 100 | $479K | 47.3% | 67.1% |
| Greed | 77 | $352K | 47.2% (slight short bias) | **68.9%** |

**Key behavioral patterns:**
- Traders are dramatically **more active during Fear** — 37% more trades, 115% more volume vs Greed days
- **Long bias flips**: traders lean long during Fear (52.2%) but slightly short during Greed (47.2%) — a contrarian rotation
- **Cross-margin usage is highest during Greed** (68.9%), suggesting traders take on more structural risk when sentiment is optimistic

### B3. Trader Segmentation (3 axes)

**Segment 1 — Cross-Margin Usage**

| Segment | Avg Total PnL | Win Rate |
|---------|---------------|----------|
| Low Cross (<30%) | $243,317 | 72% |
| Mixed (30–70%) | $342,032 | 87% |
| High Cross (>70%) | $1,293,894 | 85% |

High cross-margin users dominate total PnL — they are the most sophisticated, high-volume traders.

**Segment 2 — Trade Frequency**

Frequent traders (top 33%) generate the highest gross volume but not always the best risk-adjusted returns. Infrequent traders show lower total PnL but better median per-trade quality.

**Segment 3 — Performance Consistency**

25 of 32 accounts (78%) are profitable over the full period. The 7 loss-making accounts are concentrated in the "Opportunistic Traders" archetype (see clustering below).

---

## Key Insights (Backed by Charts)

### 📌 Insight 1 — Fear Days Have Higher Skew, Not Higher Median Returns
Fear days show the *highest average* PnL ($5,037) but *lowest median* ($104). The PnL distribution on Fear days has fat right tails — a small number of traders capture large directional moves during panic events, while most break even or lose slightly. **Greed days produce the most consistent returns** (median $236). → *Chart 1*

### 📌 Insight 2 — Traders Over-Trade During Fear (Exactly the Wrong Time)
Trade count spikes +37% and volume +115% during Fear vs Greed days. Yet median PnL is lower during Fear. The data shows traders are churning more activity with lower per-trade quality during panic, likely due to emotional reaction and stop-chasing. → *Chart 2*

### 📌 Insight 3 — Long-Ratio Diverges Counterintuitively With Sentiment
During Fear, traders *increase* long exposure (52.2% long vs 47.2% on Greed). This is a contrarian buy-the-dip behavior — and it partially works (Fear-day returns are large in mean), but is also highly volatile. The "Greed = more short" pattern is surprising and suggests these traders are fading rallies. → *Chart 5*

### 📌 Insight 4 — Cross-Margin Usage Is the Strongest Account-Level Predictor
High cross-margin users (>70% of trades) average $1.29M total PnL vs $243K for low cross-margin users. This reflects sophistication: experienced perpetuals traders use cross-margin strategically to manage portfolio-wide risk, while low cross-margin traders use isolated margin (more conservative but also lower upside). → *Chart 3*

### 📌 Insight 5 — Predictive Model Achieves ROC-AUC 0.715
A Random Forest model predicting whether a trader's *next day* will be profitable (using today's sentiment + behavioral features) achieves 0.715 ROC-AUC and 65.9% CV accuracy. Top features: **Win Rate, Total Volume, FG Index Value** — showing that behavioral consistency + sentiment together carry real predictive signal. → *Chart 8*

---

## Part C — Actionable Strategy Recommendations

### Strategy 1: "Quality Over Quantity" Protocol for Fear Days
> *"During Fear days, cap maximum daily trade count at the trader's 30-day average. Flag accounts exceeding 150% of their baseline frequency for review. Do not increase position size during Fear."*

**Rationale:** Fear days produce 37% more trades but much lower *median* PnL ($104 vs $236 on Greed). The skewed mean is driven by a few outlier wins, not consistent execution. Most of the excess trades on Fear days are noise — emotional reactions that erode P&L through fees and slippage. A frequency cap forces traders to be selective.

**Expected impact:** Reduce fee drag on Fear days (avg fee/day is highest during Fear given 115% higher volume). Protect against the liquidation risk that spikes during high-frequency panic trading.

---

### Strategy 2: Sentiment-Calibrated Position Sizing — "Scale Into Greed, Flatten Into Fear"
> *"During Greed days, allow full position sizing with standard cross-margin. During Fear days, reduce per-trade size USD by 25–30% and prefer isolated margin for new positions. Reserve larger bets for post-Fear recovery window (1–3 days after sentiment flips back above Neutral)."*

**Rationale:** Greed days show the highest *median* PnL per trader-day ($236) and the most consistent win rates (85.6%). The distribution is tight and positive — the right time to deploy full sizing. Fear days, despite high mean PnL (skewed by outliers), carry much more variance. The best risk-adjusted window is the transition *from* Fear *to* Neutral/Greed, when the contrarian long thesis (Insight 3) pays off with lower ongoing uncertainty.

**Expected impact:** Improve Sharpe ratio by concentrating capital deployment in high-median environments; reduce max drawdown by limiting size during Fear volatility spikes.

---

## Bonus

### Predictive Model
- **Algorithm:** Random Forest Classifier (200 trees, max depth 6)
- **Target:** Is next trading day profitable? (binary)
- **Features:** Sentiment encoding, FG value, # trades, win rate, long ratio, cross-margin %, avg size, total volume
- **ROC-AUC:** 0.715 | **CV Accuracy:** 65.9% ± 1.5%
- **Top predictors:** Win rate, total volume, FG index value

### KMeans Clustering — 4 Behavioral Archetypes

| Archetype | Avg PnL | Win Rate | Trades | Description |
|-----------|---------|----------|--------|-------------|
| **Elite Performers** | $1.29M | 85% | 25,370 | High-volume cross-margin pros; dominate in all conditions |
| **Disciplined Traders** | $342K | 87% | 8,219 | Moderate volume, very high win rate, large avg sizes |
| **Opportunistic Traders** | $243K | 72% | 3,783 | Low cross-margin, infrequent — includes most loss-making accounts |
| **Loss-Heavy Traders** | $169K | 93% | 4,540 | High win rate but losses when wrong are catastrophic (low avg) |

---

## Dependencies

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
scipy>=1.11
jupyter
```

---

*Primetrade.ai — Data Science Intern Round 0 Submission*
