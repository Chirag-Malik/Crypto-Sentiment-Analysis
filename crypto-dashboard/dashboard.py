import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Trader Sentiment Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Trader Performance vs Market Sentiment")
st.markdown("Analyze trading performance based on Fear & Greed market sentiment.")

# ---------------- DATA LOADING ---------------- #

DATA_DIR = Path(__file__).resolve().parent
TRADER_CSV = DATA_DIR / "historical_data.csv"
SENTIMENT_CSV = DATA_DIR / "fear_greed_index.csv"


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline().strip()

    return first_line.startswith("version https://git-lfs.github.com/spec/v1")


def sample_trader_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Timestamp IST": pd.to_datetime(
                ["01-01-2026 09:30", "02-01-2026 11:45", "03-01-2026 14:10", "04-01-2026 16:20"],
                format="%d-%m-%Y %H:%M",
            ),
            "Coin": ["BTC", "ETH", "BTC", "SOL"],
            "Closed PnL": [1500, -420, 980, 520],
        }
    )


def sample_sentiment_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
            ).date,
            "classification": ["Greed", "Fear", "Neutral", "Greed"],
            "value": [78, 23, 50, 82],
        }
    )


def load_csv(path: Path) -> pd.DataFrame:
    if is_git_lfs_pointer(path):
        st.warning(
            f"{path.name} appears to be stored with Git LFS and is not available in this environment. "
            "Using sample data instead."
        )
        if path.name == "historical_data.csv":
            return sample_trader_data()
        if path.name == "fear_greed_index.csv":
            return sample_sentiment_data()

    if not path.exists():
        st.warning(
            f"{path.name} is missing. Using sample data instead."
        )
        if path.name == "historical_data.csv":
            return sample_trader_data()
        if path.name == "fear_greed_index.csv":
            return sample_sentiment_data()

    return pd.read_csv(path)


def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_columns = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_columns:
            return lower_columns[candidate.lower()]
    return None


trader_df = load_csv(TRADER_CSV)
sentiment_df = load_csv(SENTIMENT_CSV)

# ---------------- DATA CLEANING ---------------- #

timestamp_col = find_column(
    trader_df.columns,
    ["Timestamp IST", "timestamp_ist", "Timestamp", "timestamp", "datetime", "Date"]
)
if not timestamp_col:
    st.error("Could not find a timestamp column in historical_data.csv.")
    st.stop()

trader_df["Timestamp IST"] = pd.to_datetime(
    trader_df[timestamp_col],
    errors="coerce",
    dayfirst=True,
)

if trader_df["Timestamp IST"].isna().any():
    st.warning(
        "Some timestamps could not be parsed. Check the `Timestamp IST` values in historical_data.csv."
    )

sentiment_date_col = find_column(
    sentiment_df.columns,
    ["date", "Date", "timestamp", "datetime"]
)
if not sentiment_date_col:
    st.error("Could not find a date column in fear_greed_index.csv.")
    st.stop()

sentiment_df["date"] = pd.to_datetime(
    sentiment_df[sentiment_date_col],
    errors="coerce",
).dt.date

if sentiment_df["date"].isna().any():
    st.warning(
        "Some sentiment dates could not be parsed. Check the date values in fear_greed_index.csv."
    )

trader_df["date"] = trader_df["Timestamp IST"].dt.date
sentiment_df = sentiment_df.dropna(subset=["date"])

if trader_df["date"].isna().any():
    trader_df = trader_df.dropna(subset=["date"])

merged = pd.merge(
    trader_df,
    sentiment_df,
    on="date",
    how="inner",
)

if merged.empty:
    st.warning(
        "No matching records were found between historical_data.csv and fear_greed_index.csv. "
        "Check the date formats and ensure both files cover the same date range."
    )

merged["Month"] = merged["Timestamp IST"].dt.strftime("%Y-%m")

# ---------------- SIDEBAR FILTERS ---------------- #

st.sidebar.header("Filters")

selected_coin = st.sidebar.multiselect(
    "Select Coin",
    merged["Coin"].unique() if "Coin" in merged.columns else [],
    default=merged["Coin"].unique() if "Coin" in merged.columns else [],
)

selected_sentiment = st.sidebar.multiselect(
    "Select Sentiment",
    merged["classification"].unique() if "classification" in merged.columns else [],
    default=merged["classification"].unique() if "classification" in merged.columns else [],
)

filtered_df = merged[
    (merged["Coin"].isin(selected_coin)) &
    (merged["classification"].isin(selected_sentiment))
]

if filtered_df.empty:
    st.warning("No trades match the selected filters. Adjust the filters to view results.")

# ---------------- KPI METRICS ---------------- #

total_profit = filtered_df["Closed PnL"].sum() if "Closed PnL" in filtered_df.columns else 0

total_trades = filtered_df["Closed PnL"].count() if "Closed PnL" in filtered_df.columns else 0

avg_pnl = filtered_df["Closed PnL"].mean() if "Closed PnL" in filtered_df.columns else 0

profitable_trades = (filtered_df["Closed PnL"] > 0).sum() if "Closed PnL" in filtered_df.columns else 0
loss_trades = (filtered_df["Closed PnL"] < 0).sum() if "Closed PnL" in filtered_df.columns else 0

win_rate = 0.0
if profitable_trades + loss_trades:
    win_rate = (profitable_trades / (profitable_trades + loss_trades)) * 100

# ---------------- KPI CARDS ---------------- #

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Profit", f"${total_profit:,.0f}")
col2.metric("Total Trades", f"{total_trades}")
col3.metric("Average PnL", f"${avg_pnl:,.2f}")
col4.metric("Win Rate", f"{win_rate:.2f}%")

st.divider()

# ---------------- CHARTS ---------------- #

profit_by_senti = (
    filtered_df.groupby("classification")["Closed PnL"].sum()
    if "classification" in filtered_df.columns and "Closed PnL" in filtered_df.columns
    else pd.Series(dtype="float64")
)

avg_profit = (
    filtered_df.groupby("classification")["Closed PnL"].mean()
    if "classification" in filtered_df.columns and "Closed PnL" in filtered_df.columns
    else pd.Series(dtype="float64")
)

st.subheader("Profit by Sentiment")
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    profit_by_senti.plot(kind="bar", ax=ax)
    ax.set_ylabel("Profit")
    st.pyplot(fig)

with col2:
    st.subheader("Average Profit by Sentiment")
    fig, ax = plt.subplots()
    avg_profit.plot(kind="bar", ax=ax)
    ax.set_ylabel("Average PnL")
    st.pyplot(fig)

# ---------------- MONTHLY PROFIT ---------------- #

st.subheader("Monthly Profit Trend")
monthly_profit = (
    filtered_df.groupby("Month")["Closed PnL"].sum()
    if "Month" in filtered_df.columns and "Closed PnL" in filtered_df.columns
    else pd.Series(dtype="float64")
)

fig, ax = plt.subplots(figsize=(10, 4))
monthly_profit.plot(kind="line", marker="o", ax=ax)
ax.set_ylabel("Profit")
st.pyplot(fig)

# ---------------- TOP COINS ---------------- #

st.subheader("Top Performing Coins")
coin_profit = (
    filtered_df.groupby("Coin")["Closed PnL"].sum().sort_values(ascending=False).head(10)
    if "Coin" in filtered_df.columns and "Closed PnL" in filtered_df.columns
    else pd.Series(dtype="float64")
)

fig, ax = plt.subplots(figsize=(10, 5))
coin_profit.plot(kind="bar", ax=ax)
ax.set_ylabel("Profit")
st.pyplot(fig)

# ---------------- RISK ANALYSIS ---------------- #

st.subheader("Risk Analysis by Sentiment")
risk = (
    filtered_df.groupby("classification")["Closed PnL"].std()
    if "classification" in filtered_df.columns and "Closed PnL" in filtered_df.columns
    else pd.Series(dtype="float64")
)

st.dataframe(
    risk.reset_index().rename(columns={"Closed PnL": "Risk (Std Dev)"})
)

# ---------------- INSIGHTS ---------------- #

st.subheader("Key Insights")

best_sentiment = profit_by_senti.idxmax() if not profit_by_senti.empty else "N/A"
worst_sentiment = profit_by_senti.idxmin() if not profit_by_senti.empty else "N/A"
best_coin = coin_profit.idxmax() if not coin_profit.empty else "N/A"

st.success(
    f"""
    ✅ Traders performed best during **{best_sentiment}** sentiment.

    📉 Worst performance occurred during **{worst_sentiment}** sentiment.

    🪙 Most profitable coin: **{best_coin}**

    📊 Overall win rate is **{win_rate:.2f}%**
    """
)

# ---------------- RAW DATA ---------------- #

with st.expander("View Raw Data"):
    st.dataframe(filtered_df)

# ---------------- DOWNLOAD BUTTON ---------------- #

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download Filtered Data",
    data=csv,
    file_name="filtered_trading_data.csv",
    mime="text/csv"
)
