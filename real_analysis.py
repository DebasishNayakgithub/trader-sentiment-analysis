"""
Primetrade.ai – Data Science Intern Assignment (REAL DATA)
Trader Performance vs Market Sentiment
Datasets: fear_greed_index.csv  ×  historical_data_csv.gz
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ── Palette ──────────────────────────────────────────────────────────────────
FEAR_COLOR    = "#E63946"
GREED_COLOR   = "#2EC4B6"
NEUTRAL_COLOR = "#F4A261"
EF_COLOR      = "#9B1D20"   # Extreme Fear
EG_COLOR      = "#1A7A74"   # Extreme Greed
BG   = "#0D1117"
CARD = "#161B22"
TEXT = "#E6EDF3"
GRID = "#30363D"
ACCENT = "#A8DADC"

plt.rcParams.update({
    "figure.facecolor": BG,   "axes.facecolor":   CARD,
    "axes.edgecolor":   GRID, "axes.labelcolor":  TEXT,
    "xtick.color":      TEXT, "ytick.color":      TEXT,
    "text.color":       TEXT, "grid.color":       GRID,
    "grid.linestyle":   "--", "grid.alpha":       0.4,
    "legend.facecolor": CARD, "legend.edgecolor": GRID,
    "axes.titlesize":   13,   "axes.labelsize":   11,
    "font.family":      "DejaVu Sans",
})

OUT = "/home/claude/primetrade_project/charts"

SENT_COLORS = {
    "Extreme Fear": EF_COLOR, "Fear": FEAR_COLOR,
    "Neutral": NEUTRAL_COLOR,
    "Greed": GREED_COLOR,    "Extreme Greed": EG_COLOR,
}
BIN_COLORS = {"Fear": FEAR_COLOR, "Greed": GREED_COLOR, "Neutral": NEUTRAL_COLOR}

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD & CLEAN
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  PRIMETRADE.AI — REAL DATA ANALYSIS")
print("=" * 65)

# ── Fear/Greed ───────────────────────────────────────────────────────────────
fg_raw = pd.read_csv("/mnt/user-data/uploads/fear_greed_index.csv")
fg_raw["date"] = pd.to_datetime(fg_raw["date"])
fg_raw = fg_raw.sort_values("date").drop_duplicates("date")
fg_raw["sentiment"] = fg_raw["classification"].apply(
    lambda x: "Fear" if "Fear" in x else ("Greed" if "Greed" in x else "Neutral"))

# ── Trader data ──────────────────────────────────────────────────────────────
td_raw = pd.read_csv("/mnt/user-data/uploads/historical_data_csv.gz", compression="gzip")
td_raw["datetime"] = pd.to_datetime(td_raw["Timestamp IST"], format="%d-%m-%Y %H:%M", dayfirst=True)
td_raw["date"]     = td_raw["datetime"].dt.normalize()

print(f"\n[RAW] Fear/Greed : {fg_raw.shape[0]:,} rows | {fg_raw['date'].min().date()} → {fg_raw['date'].max().date()}")
print(f"[RAW] Trader data: {td_raw.shape[0]:,} rows | {td_raw['date'].min().date()} → {td_raw['date'].max().date()}")
print(f"      Unique accounts : {td_raw['Account'].nunique()}")
print(f"      Unique coins    : {td_raw['Coin'].nunique()}")

# ── Missing / duplicates ─────────────────────────────────────────────────────
print("\n[QC] Missing values — Fear/Greed:")
print(fg_raw.isnull().sum().to_string())
print("\n[QC] Missing values — Trader:")
print(td_raw.isnull().sum().to_string())
print(f"\n[QC] Duplicate rows — FG : {fg_raw.duplicated().sum()}")
print(f"[QC] Duplicate rows — TD : {td_raw.duplicated().sum()}")

# ── Filter to overlap window ─────────────────────────────────────────────────
overlap_start = max(td_raw["date"].min(), fg_raw["date"].min())
overlap_end   = min(td_raw["date"].max(), fg_raw["date"].max())

fg = fg_raw[(fg_raw["date"] >= overlap_start) & (fg_raw["date"] <= overlap_end)].copy()
td = td_raw[(td_raw["date"] >= overlap_start) & (td_raw["date"] <= overlap_end)].copy()

print(f"\n[OVERLAP] {overlap_start.date()} → {overlap_end.date()}")
print(f"  FG rows: {fg.shape[0]:,}  |  TD rows: {td.shape[0]:,}")

# ── Merge ────────────────────────────────────────────────────────────────────
merged = td.merge(
    fg[["date","value","classification","sentiment"]].rename(columns={"value":"fg_value"}),
    on="date", how="left"
)
print(f"  Merged shape: {merged.shape}  | Unmatched: {merged['sentiment'].isna().sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[FEAT] Engineering features …")

# Identify closing trades (where PnL is realized)
# Directions that close a position
close_dirs = {"Close Long", "Close Short", "Long > Short", "Short > Long",
              "Liquidated Isolated Short", "Liquidated Isolated Long",
              "Auto-Deleveraging"}
merged["is_close"] = merged["Direction"].isin(close_dirs)
merged["is_long"]  = merged["Direction"].isin({"Open Long", "Close Short", "Long > Short"}) | \
                     (merged["Side"] == "BUY")
merged["pnl_net"]  = merged["Closed PnL"] - merged["Fee"]   # net of fees
merged["is_win"]   = (merged["Closed PnL"] > 0).astype(int)

# ── Daily per-account aggregates ─────────────────────────────────────────────
def daily_agg(df):
    closes = df[df["is_close"] | (df["Closed PnL"] != 0)]
    opens  = df[~df["is_close"]]
    return pd.Series({
        "n_trades"        : len(df),
        "n_closes"        : len(closes),
        "daily_pnl"       : df["pnl_net"].sum(),
        "realized_pnl"    : closes["Closed PnL"].sum(),
        "total_fees"      : df["Fee"].sum(),
        "avg_size_usd"    : df["Size USD"].mean(),
        "total_vol_usd"   : df["Size USD"].sum(),
        "wins"            : (closes["Closed PnL"] > 0).sum(),
        "n_longs"         : df["is_long"].sum(),
        "n_shorts"        : (~df["is_long"]).sum(),
        "cross_margin_pct": df["Crossed"].mean(),
        "n_liquidations"  : (df["Direction"].str.contains("Liquidat", na=False)).sum(),
    })

daily = (merged.groupby(["date","Account","sentiment","classification","fg_value"])
               .apply(daily_agg, include_groups=False)
               .reset_index())

daily["win_rate"]         = daily["wins"] / daily["n_closes"].replace(0, np.nan)
daily["long_ratio"]       = daily["n_longs"] / (daily["n_trades"] + 1e-9)
daily["is_day_winner"]    = (daily["daily_pnl"] > 0).astype(int)
daily["pnl_per_trade"]    = daily["daily_pnl"] / daily["n_trades"].replace(0, np.nan)

# ── Account-level aggregates ──────────────────────────────────────────────────
def account_agg(df):
    closes = df[df["is_close"] | (df["Closed PnL"] != 0)]
    pnl_cs = closes.sort_values("datetime")["Closed PnL"].cumsum()
    dd     = (pnl_cs - pnl_cs.cummax()).min() if len(pnl_cs) else 0
    return pd.Series({
        "total_pnl"      : df["pnl_net"].sum(),
        "realized_pnl"   : closes["Closed PnL"].sum(),
        "total_fees"     : df["Fee"].sum(),
        "n_trades"       : len(df),
        "n_closes"       : len(closes),
        "win_rate"       : (closes["Closed PnL"] > 0).mean() if len(closes) else np.nan,
        "avg_size_usd"   : df["Size USD"].mean(),
        "total_vol_usd"  : df["Size USD"].sum(),
        "cross_margin_pct": df["Crossed"].mean(),
        "long_ratio"     : df["is_long"].mean(),
        "max_drawdown"   : dd,
        "n_liquidations" : (df["Direction"].str.contains("Liquidat", na=False)).sum(),
        "n_coins"        : df["Coin"].nunique(),
        "active_days"    : df["date"].nunique(),
    })

acct = (merged.groupby("Account")
              .apply(account_agg, include_groups=False)
              .reset_index())

# Segments
acct["cross_segment"] = acct["cross_margin_pct"].apply(
    lambda x: "High Cross (>70%)" if x > 0.7 else ("Low Cross (<30%)" if x < 0.3 else "Mixed"))
acct["freq_segment"] = pd.qcut(acct["n_trades"], q=3,
                                labels=["Infrequent","Moderate","Frequent"])
acct["perf_segment"] = acct["total_pnl"].apply(
    lambda x: "Consistent Winner" if x > 0 else "Consistent Loser")
acct["size_segment"] = pd.qcut(acct["avg_size_usd"], q=3,
                                labels=["Small","Medium","Large"])

print(f"  Daily records: {daily.shape[0]:,}")
print(f"  Account records: {acct.shape[0]}")
print(f"\n  Account-level PnL summary:")
print(acct[["total_pnl","win_rate","n_trades","max_drawdown"]].describe().round(2).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# B-1: PERFORMANCE — FEAR vs GREED
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[B1] Performance by Sentiment …")

sent_perf = daily.groupby("sentiment").agg(
    n_trader_days    = ("daily_pnl","count"),
    avg_daily_pnl    = ("daily_pnl","mean"),
    median_daily_pnl = ("daily_pnl","median"),
    total_pnl        = ("daily_pnl","sum"),
    win_rate         = ("win_rate","mean"),
    avg_n_trades     = ("n_trades","mean"),
    avg_vol_usd      = ("total_vol_usd","mean"),
    avg_long_ratio   = ("long_ratio","mean"),
    liquidation_days = ("n_liquidations", lambda x: (x>0).sum()),
).round(3)
print("\n  Sentiment Performance Table:")
print(sent_perf.to_string())

# 5-class breakdown
cls_perf = daily.groupby("classification").agg(
    avg_daily_pnl  = ("daily_pnl","mean"),
    win_rate       = ("win_rate","mean"),
    avg_n_trades   = ("n_trades","mean"),
).round(3).reindex(["Extreme Fear","Fear","Neutral","Greed","Extreme Greed"])
print("\n  5-class Classification Performance:")
print(cls_perf.to_string())

fear_pnl  = daily.loc[daily["sentiment"]=="Fear",  "daily_pnl"]
greed_pnl = daily.loc[daily["sentiment"]=="Greed", "daily_pnl"]
u, p = stats.mannwhitneyu(fear_pnl, greed_pnl, alternative="two-sided")
print(f"\n  Mann-Whitney U={u:.0f}, p={p:.6f} → Significant: {p<0.05}")

# ═══════════════════════════════════════════════════════════════════════════════
# B-2: BEHAVIOR SHIFTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[B2] Behavioral shifts …")
behav = daily.groupby("sentiment").agg(
    avg_n_trades    = ("n_trades","mean"),
    avg_vol_usd     = ("total_vol_usd","mean"),
    avg_long_ratio  = ("long_ratio","mean"),
    cross_margin    = ("cross_margin_pct","mean"),
    avg_size_usd    = ("avg_size_usd","mean"),
).round(3)
print("\n  Behavior by Sentiment:")
print(behav.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — PnL distribution Fear vs Greed + 5-class win rate
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 6), facecolor=BG)
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2])

# Histogram
for s, col in [("Fear", FEAR_COLOR), ("Greed", GREED_COLOR), ("Neutral", NEUTRAL_COLOR)]:
    data = daily.loc[daily["sentiment"]==s, "daily_pnl"]
    ax1.hist(data.clip(-5000, 5000), bins=60, alpha=0.72, color=col,
             label=f"{s}  (μ={data.mean():.0f}$)", edgecolor="none")
ax1.axvline(0, color="white", lw=1.2, ls="--", alpha=0.7)
ax1.set_xlabel("Daily Net PnL per Trader (USD)")
ax1.set_ylabel("Frequency")
ax1.set_title("Daily PnL Distribution — Fear vs Greed Days", fontweight="bold")
ax1.legend(); ax1.grid(True)

# 5-class win rate bar
order5 = ["Extreme Fear","Fear","Neutral","Greed","Extreme Greed"]
wr5 = daily.groupby("classification")["win_rate"].mean().reindex(order5)
cols5 = [SENT_COLORS[c] for c in order5]
bars = ax2.bar(range(len(order5)), wr5.values, color=cols5, edgecolor="none", width=0.6)
ax2.set_xticks(range(len(order5)))
ax2.set_xticklabels(["Ext.\nFear","Fear","Neutral","Greed","Ext.\nGreed"], fontsize=9)
ax2.axhline(0.5, color="white", lw=1, ls="--", alpha=0.7, label="50%")
ax2.set_ylabel("Avg Win Rate")
ax2.set_title("Win Rate by Classification", fontweight="bold")
ax2.set_ylim(0, 0.75); ax2.legend(); ax2.grid(True, axis="y")
for bar, val in zip(bars, wr5.values):
    if not np.isnan(val):
        ax2.text(bar.get_x()+bar.get_width()/2, val+0.01,
                 f"{val:.1%}", ha="center", fontsize=9, fontweight="bold")

fig.suptitle("Chart 1 — Trader Performance: Fear vs Greed Days  (Real Hyperliquid Data)",
             fontsize=14, fontweight="bold", color=TEXT, y=1.02)
plt.savefig(f"{OUT}/chart1_pnl_winrate.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("  [Chart 1 saved]")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Behavior Shifts: volume, trades, long-ratio, cross-margin
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), facecolor=BG)
fig.suptitle("Chart 2 — Trader Behavior Shifts by Sentiment (Real Data)",
             fontsize=14, fontweight="bold", color=TEXT, y=1.01)

order3 = ["Fear","Neutral","Greed"]
cols3  = [BIN_COLORS[s] for s in order3]
metrics_behav = [
    ("avg_n_trades",   "Avg Trades / Day",      False),
    ("avg_vol_usd",    "Avg Daily Volume (USD)", False),
    ("avg_long_ratio", "Avg Long Ratio",         True),
    ("cross_margin",   "Avg Cross-Margin %",     True),
]

for ax, (col, label, pct) in zip(axes, metrics_behav):
    vals = behav.loc[order3, col]
    bars = ax.bar(order3, vals.values, color=cols3, edgecolor="none", width=0.5)
    ax.set_title(label, fontweight="bold")
    ax.grid(True, axis="y")
    if pct:
        ax.axhline(0.5, color="white", lw=0.8, ls="--", alpha=0.6)
        ax.set_ylim(0, 0.85)
    for bar, val in zip(bars, vals.values):
        fmt = f"{val:.1%}" if pct else f"{val:,.0f}"
        ax.text(bar.get_x()+bar.get_width()/2,
                val + vals.max()*0.02, fmt,
                ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/chart2_behavior_shifts.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("  [Chart 2 saved]")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Account-level PnL vs win rate scatter + segments
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
fig.suptitle("Chart 3 — Trader Segments: PnL, Win Rate & Drawdown",
             fontsize=14, fontweight="bold", color=TEXT, y=1.01)

# Scatter: total PnL vs win rate, sized by volume
sc = axes[0].scatter(
    acct["win_rate"], acct["total_pnl"],
    s=np.clip(acct["total_vol_usd"]/2e5, 30, 400),
    c=acct["total_pnl"], cmap="RdYlGn",
    alpha=0.85, edgecolors="none",
    vmin=acct["total_pnl"].quantile(0.05),
    vmax=acct["total_pnl"].quantile(0.95),
)
plt.colorbar(sc, ax=axes[0], label="Total PnL")
axes[0].axhline(0, color="white", lw=0.8, ls="--")
axes[0].axvline(0.5, color="white", lw=0.8, ls="--")
axes[0].set_xlabel("Win Rate"); axes[0].set_ylabel("Total PnL (USD)")
axes[0].set_title("All 32 Accounts\n(size = trading volume)", fontweight="bold")
axes[0].grid(True)

# Cross-margin segment PnL
csegs = acct.groupby("cross_segment")["total_pnl"].mean().sort_values()
colors_cs = [FEAR_COLOR if v < 0 else GREED_COLOR for v in csegs.values]
axes[1].barh(csegs.index, csegs.values, color=colors_cs, edgecolor="none", height=0.45)
axes[1].axvline(0, color="white", lw=0.8, ls="--")
axes[1].set_xlabel("Avg Total PnL (USD)")
axes[1].set_title("Avg PnL by Cross-Margin Usage", fontweight="bold")
axes[1].grid(True, axis="x")
for i, (idx, val) in enumerate(csegs.items()):
    axes[1].text(val + (abs(csegs).max()*0.02 * (1 if val >= 0 else -1)),
                 i, f"${val:,.0f}", va="center", fontsize=9)

# Max drawdown vs PnL
axes[2].scatter(acct["max_drawdown"], acct["total_pnl"],
                c=[GREED_COLOR if p > 0 else FEAR_COLOR for p in acct["total_pnl"]],
                s=80, alpha=0.85, edgecolors="none")
axes[2].axhline(0, color="white", lw=0.8, ls="--")
axes[2].set_xlabel("Max Drawdown (USD)")
axes[2].set_ylabel("Total PnL (USD)")
axes[2].set_title("Max Drawdown vs Total PnL", fontweight="bold")
axes[2].grid(True)

plt.tight_layout()
plt.savefig(f"{OUT}/chart3_account_segments.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("  [Chart 3 saved]")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Time-series: FG index + market PnL rolling
# ═══════════════════════════════════════════════════════════════════════════════
daily_mkt = daily.groupby(["date","fg_value","classification","sentiment"]).agg(
    total_pnl   = ("daily_pnl","sum"),
    n_trades    = ("n_trades","sum"),
    long_ratio  = ("long_ratio","mean"),
).reset_index().sort_values("date")

daily_mkt["pnl_7d"] = daily_mkt["total_pnl"].rolling(7, min_periods=1).mean()
daily_mkt["pnl_30d"]= daily_mkt["total_pnl"].rolling(30, min_periods=1).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), facecolor=BG,
                                 gridspec_kw={"height_ratios":[2,1],"hspace":0.08})
fig.suptitle("Chart 4 — Fear/Greed Index vs Market PnL (May 2023 – May 2025)",
             fontsize=14, fontweight="bold", color=TEXT, y=0.99)

# Top: FG index
ax1.plot(daily_mkt["date"], daily_mkt["fg_value"], color=NEUTRAL_COLOR, lw=1.4, label="FG Index")
ax1.fill_between(daily_mkt["date"], daily_mkt["fg_value"], 50,
                  where=daily_mkt["fg_value"] >= 50, alpha=0.18, color=GREED_COLOR, label="Greed zone")
ax1.fill_between(daily_mkt["date"], daily_mkt["fg_value"], 50,
                  where=daily_mkt["fg_value"] < 50,  alpha=0.18, color=FEAR_COLOR,  label="Fear zone")
ax1.axhline(50, color=GRID, lw=0.8)
ax1.axhline(25, color=EF_COLOR, lw=0.6, ls=":")
ax1.axhline(75, color=EG_COLOR, lw=0.6, ls=":")
ax1.set_ylabel("Fear/Greed Index Value"); ax1.set_ylim(0, 100)
ax1.legend(loc="upper left"); ax1.grid(True); ax1.set_xticklabels([])

# Bottom: Market PnL
ax2b = ax2.twinx()
colors_bar = [BIN_COLORS.get(s, NEUTRAL_COLOR) for s in daily_mkt["sentiment"]]
ax2.bar(daily_mkt["date"], daily_mkt["total_pnl"], color=colors_bar, alpha=0.4, width=0.8, label="Daily Market PnL")
ax2b.plot(daily_mkt["date"], daily_mkt["pnl_30d"], color=ACCENT, lw=2, label="30d MA PnL")
ax2.set_ylabel("Daily Market PnL (USD)"); ax2b.set_ylabel("30d MA (USD)", color=ACCENT)
ax2b.tick_params(axis="y", labelcolor=ACCENT)
lines2, lbl2 = ax2.get_legend_handles_labels()
lines2b, lbl2b = ax2b.get_legend_handles_labels()
ax2.legend(lines2+lines2b, lbl2+lbl2b, loc="upper left"); ax2.grid(True)

plt.savefig(f"{OUT}/chart4_timeseries.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("  [Chart 4 saved]")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Long ratio & volume heatmap by month × sentiment
# ═══════════════════════════════════════════════════════════════════════════════
daily["month"]  = daily["date"].dt.to_period("M").astype(str)
daily["ym"]     = daily["date"].dt.to_period("M")

monthly_sent = daily.groupby(["ym","sentiment"]).agg(
    avg_pnl      = ("daily_pnl","mean"),
    win_rate     = ("win_rate","mean"),
    long_ratio   = ("long_ratio","mean"),
    n_obs        = ("daily_pnl","count"),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), facecolor=BG)
fig.suptitle("Chart 5 — Long-Ratio Shift & Coin Mix by Sentiment",
             fontsize=14, fontweight="bold", color=TEXT, y=1.01)

# Long ratio by sentiment (violin)
order3 = ["Fear","Neutral","Greed"]
data_lr = [daily.loc[daily["sentiment"]==s, "long_ratio"].dropna() for s in order3]
vp = axes[0].violinplot(data_lr, showmedians=True, showextrema=True)
for body, col in zip(vp["bodies"], [FEAR_COLOR, NEUTRAL_COLOR, GREED_COLOR]):
    body.set_facecolor(col); body.set_alpha(0.75)
vp["cmedians"].set_color("white"); vp["cbars"].set_color(GRID)
vp["cmins"].set_color(GRID);       vp["cmaxes"].set_color(GRID)
axes[0].set_xticks([1,2,3]); axes[0].set_xticklabels(order3)
axes[0].axhline(0.5, color="white", lw=0.8, ls="--", alpha=0.6, label="50% neutral")
axes[0].set_ylabel("Long Ratio (fraction of buys)")
axes[0].set_title("Long Ratio Distribution by Sentiment", fontweight="bold")
axes[0].legend(); axes[0].grid(True, axis="y")

# Top coins by sentiment PnL
top_coins = merged.groupby(["sentiment","Coin"])["pnl_net"].sum().reset_index()
top_by_sent = (top_coins.groupby("sentiment", group_keys=True)
                         .apply(lambda x: x.nlargest(5, "pnl_net"))
                         .reset_index(level=0)
                         .reset_index(drop=True))

sent_order = ["Fear","Neutral","Greed"]
bar_w = 0.25; x = np.arange(5)
for i, (sent, col) in enumerate(zip(sent_order, [FEAR_COLOR, NEUTRAL_COLOR, GREED_COLOR])):
    sub = top_by_sent[top_by_sent["sentiment"]==sent].head(5)
    axes[1].bar(x + i*bar_w, sub["pnl_net"].values, width=bar_w,
                label=sent, color=col, alpha=0.85, edgecolor="none")
axes[1].set_xticks(x + bar_w)
greed_coins = top_by_sent[top_by_sent["sentiment"]=="Greed"]["Coin"].head(5).tolist()
axes[1].set_xticklabels(greed_coins, fontsize=9)
axes[1].set_ylabel("Total Net PnL (USD)")
axes[1].set_title("Top 5 Coin PnL by Sentiment Period", fontweight="bold")
axes[1].axhline(0, color="white", lw=0.8, ls="--")
axes[1].legend(); axes[1].grid(True, axis="y")

plt.tight_layout()
plt.savefig(f"{OUT}/chart5_long_ratio_coins.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("  [Chart 5 saved]")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Trader clustering (KMeans k=4)
# ═══════════════════════════════════════════════════════════════════════════════
feat_cols = ["total_pnl","win_rate","n_trades","avg_size_usd",
             "cross_margin_pct","long_ratio","max_drawdown","total_vol_usd"]
X  = acct[feat_cols].fillna(0)
Xs = StandardScaler().fit_transform(X)

km = KMeans(n_clusters=4, random_state=42, n_init=20)
acct["cluster"] = km.fit_predict(Xs)

# Label clusters by descending total PnL
pnl_rank = acct.groupby("cluster")["total_pnl"].mean().sort_values(ascending=False)
cnames = {c: n for c, n in zip(pnl_rank.index,
          ["Elite Performers","Disciplined Traders","Opportunistic Traders","Loss-Heavy Traders"])}
acct["archetype"] = acct["cluster"].map(cnames)

print("\n  Cluster summary:")
cl_summary = acct.groupby("archetype")[feat_cols[:6]].mean().round(2)
print(cl_summary.to_string())

arch_cols = ["#2EC4B6","#F4A261","#8338EC","#E63946"]
arch_order = ["Elite Performers","Disciplined Traders","Opportunistic Traders","Loss-Heavy Traders"]
col_map = {n: c for n, c in zip(arch_order, arch_cols)}

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
fig.suptitle("Chart 6 — Trader Behavioral Archetypes (KMeans k=4)",
             fontsize=14, fontweight="bold", color=TEXT, y=1.01)

for arch in arch_order:
    sub = acct[acct["archetype"]==arch]
    axes[0].scatter(sub["win_rate"], sub["total_pnl"],
                    s=np.clip(sub["total_vol_usd"]/5e5, 40, 400),
                    color=col_map[arch], label=arch, alpha=0.85, edgecolors="none")
axes[0].axhline(0, color="white", lw=0.8, ls="--")
axes[0].axvline(0.5, color="white", lw=0.8, ls="--")
axes[0].set_xlabel("Win Rate"); axes[0].set_ylabel("Total PnL (USD)")
axes[0].set_title("Archetype Map: Win Rate × Total PnL", fontweight="bold")
axes[0].legend(fontsize=9); axes[0].grid(True)

# Radar-style bar
norm_feats = ["win_rate","n_trades","avg_size_usd","cross_margin_pct","long_ratio"]
cat_means  = acct.groupby("archetype")[norm_feats].mean()
cat_norm   = (cat_means - cat_means.min()) / (cat_means.max() - cat_means.min() + 1e-9)
x = np.arange(len(norm_feats)); w = 0.18
for i, arch in enumerate(arch_order):
    row = cat_norm.loc[arch] if arch in cat_norm.index else pd.Series([0]*len(norm_feats))
    axes[1].bar(x+i*w, row.values, width=w, label=arch, color=arch_cols[i], alpha=0.85, edgecolor="none")
axes[1].set_xticks(x+w*1.5)
axes[1].set_xticklabels(["Win\nRate","# Trades","Avg Size","Cross\nMargin","Long\nRatio"], fontsize=9)
axes[1].set_title("Normalized Behavioral Profile by Archetype", fontweight="bold")
axes[1].legend(fontsize=8); axes[1].grid(True, axis="y")

plt.tight_layout()
plt.savefig(f"{OUT}/chart6_clusters.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("  [Chart 6 saved]")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Sentiment × account-pair: who wins in Fear vs Greed
# ═══════════════════════════════════════════════════════════════════════════════
# Per-account per-sentiment aggregation
acct_sent = daily.groupby(["Account","sentiment"]).agg(
    avg_pnl   = ("daily_pnl","mean"),
    win_rate  = ("win_rate","mean"),
    n_days    = ("daily_pnl","count"),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), facecolor=BG)
fig.suptitle("Chart 7 — Per-Account PnL: Fear vs Greed Breakdown",
             fontsize=14, fontweight="bold", color=TEXT, y=1.01)

# Heatmap: account × sentiment avg PnL
pivot = acct_sent.pivot(index="Account", columns="sentiment", values="avg_pnl").fillna(0)
pivot = pivot[["Fear","Neutral","Greed"]]
pivot_short = pivot.copy()
pivot_short.index = [a[:10]+"…" for a in pivot_short.index]

im = axes[0].imshow(pivot_short.values, cmap="RdYlGn", aspect="auto",
                     vmin=-500, vmax=500)
plt.colorbar(im, ax=axes[0], label="Avg Daily PnL (USD)", shrink=0.8)
axes[0].set_xticks([0,1,2]); axes[0].set_xticklabels(["Fear","Neutral","Greed"])
axes[0].set_yticks(range(len(pivot_short))); axes[0].set_yticklabels(pivot_short.index, fontsize=7)
axes[0].set_title("Avg Daily PnL Heatmap\n(Account × Sentiment)", fontweight="bold")

# Delta: Greed PnL − Fear PnL per account
delta = (pivot["Greed"] - pivot["Fear"]).sort_values()
bar_cols = [GREED_COLOR if v >= 0 else FEAR_COLOR for v in delta.values]
axes[1].barh([a[:10]+"…" for a in delta.index], delta.values,
              color=bar_cols, edgecolor="none", height=0.65)
axes[1].axvline(0, color="white", lw=0.8, ls="--")
axes[1].set_xlabel("ΔPnL = Greed_avg − Fear_avg (USD/day)")
axes[1].set_title("Each Account's Sentiment Edge\n(Greed − Fear avg daily PnL)", fontweight="bold")
axes[1].grid(True, axis="x")

plt.tight_layout()
plt.savefig(f"{OUT}/chart7_per_account_sentiment.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("  [Chart 7 saved]")

# ═══════════════════════════════════════════════════════════════════════════════
# BONUS — Random Forest Predictive Model
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[BONUS] Predictive Model …")

daily_m = daily.copy().sort_values(["Account","date"])
daily_m["next_win"]     = daily_m.groupby("Account")["is_day_winner"].shift(-1)
daily_m["next_pnl"]     = daily_m.groupby("Account")["daily_pnl"].shift(-1)
daily_m["fg_val"]       = daily_m["fg_value"]
daily_m = daily_m.dropna(subset=["next_win"])

le = LabelEncoder()
daily_m["sent_enc"] = le.fit_transform(daily_m["sentiment"])

feat_model = ["sent_enc","fg_val","n_trades","win_rate","long_ratio",
              "cross_margin_pct","avg_size_usd","total_vol_usd"]
Xm = daily_m[feat_model].fillna(0)
ym = daily_m["next_win"].astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(Xm, ym, test_size=0.2, random_state=42, stratify=ym)
rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42)
rf.fit(X_tr, y_tr)
y_pred = rf.predict(X_te)
y_prob = rf.predict_proba(X_te)[:,1]

cv = cross_val_score(rf, Xm, ym, cv=StratifiedKFold(5, shuffle=True, random_state=42))
print(f"  ROC-AUC     : {roc_auc_score(y_te, y_prob):.4f}")
print(f"  CV Accuracy : {cv.mean():.4f} ± {cv.std():.4f}")
print("\n  Classification Report:")
print(classification_report(y_te, y_pred, target_names=["Loss Day","Profit Day"]))

# Feature importance chart
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
imp = pd.Series(rf.feature_importances_, index=feat_model).sort_values(ascending=True)
display_names = {"sent_enc":"Sentiment","fg_val":"FG Index Value",
                 "n_trades":"# Trades","win_rate":"Win Rate",
                 "long_ratio":"Long Ratio","cross_margin_pct":"Cross Margin %",
                 "avg_size_usd":"Avg Trade Size","total_vol_usd":"Total Volume"}
imp.index = [display_names.get(i, i) for i in imp.index]
bar_colors = [GREED_COLOR if v >= imp.median() else ACCENT for v in imp.values]
bars = ax.barh(imp.index, imp.values, color=bar_colors, edgecolor="none", height=0.55)
ax.set_title("Chart 8 — Feature Importance: Predicting Next-Day Profitability",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score"); ax.grid(True, axis="x")
for bar, val in zip(bars, imp.values):
    ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/chart8_feature_importance.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("  [Chart 8 saved]")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY PRINTOUT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  FINAL KEY NUMBERS")
print("="*65)
for s in ["Fear","Neutral","Greed"]:
    sub = daily[daily["sentiment"]==s]
    print(f"\n  {s} ({len(sub):,} trader-days):")
    print(f"    Avg daily PnL   : ${sub['daily_pnl'].mean():>10.2f}")
    print(f"    Median daily PnL: ${sub['daily_pnl'].median():>10.2f}")
    print(f"    Avg win rate    : {sub['win_rate'].mean():>10.2%}")
    print(f"    Avg # trades    : {sub['n_trades'].mean():>10.2f}")
    print(f"    Avg long ratio  : {sub['long_ratio'].mean():>10.2%}")
    print(f"    Avg vol USD     : ${sub['total_vol_usd'].mean():>10.0f}")

print(f"\n  Mann-Whitney p (Fear vs Greed PnL): {p:.6f}")
print(f"\n  Model ROC-AUC: {roc_auc_score(y_te, y_prob):.4f}")
print("\n  Top 5 accounts by total PnL:")
print(acct.nlargest(5,"total_pnl")[["Account","total_pnl","win_rate","n_trades","archetype"]].to_string(index=False))
print("\n  Bottom 5 accounts by total PnL:")
print(acct.nsmallest(5,"total_pnl")[["Account","total_pnl","win_rate","n_trades","archetype"]].to_string(index=False))

print("\n✅  All charts saved. Analysis complete.")
