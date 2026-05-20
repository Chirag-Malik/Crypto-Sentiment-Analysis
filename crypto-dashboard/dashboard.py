import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Trader Sentiment Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Trader Performance vs Market Sentiment")
st.markdown("Analyze trading performance based on Fear & Greed market sentiment.")

# ---------------- LOAD DATA ---------------- #

trader_df = pd.read_csv("historical_data.csv")
sentiment_df = pd.read_csv("fear_greed_index.csv")

# ---------------- DATA CLEANING ---------------- #

trader_df['Timestamp IST'] = pd.to_datetime(
    trader_df['Timestamp IST'],
    format='%d-%m-%Y %H:%M'
)

sentiment_df['date'] = pd.to_datetime(
    sentiment_df['date'],
    format='%Y-%m-%d'
)

trader_df['date'] = trader_df['Timestamp IST'].dt.date
sentiment_df['date'] = sentiment_df['date'].dt.date

merged = pd.merge(
    trader_df,
    sentiment_df,
    on='date',
    how='inner'
)

merged['Month'] = merged['Timestamp IST'].dt.strftime('%Y-%m')

# ---------------- SIDEBAR FILTERS ---------------- #

st.sidebar.header("Filters")

selected_coin = st.sidebar.multiselect(
    "Select Coin",
    merged['Coin'].unique(),
    default=merged['Coin'].unique()
)

selected_sentiment = st.sidebar.multiselect(
    "Select Sentiment",
    merged['classification'].unique(),
    default=merged['classification'].unique()
)

filtered_df = merged[
    (merged['Coin'].isin(selected_coin)) &
    (merged['classification'].isin(selected_sentiment))
]

# ---------------- KPI METRICS ---------------- #

total_profit = filtered_df['Closed PnL'].sum()
total_trades = filtered_df['Closed PnL'].count()
avg_pnl = filtered_df['Closed PnL'].mean()

profitable_trades = (filtered_df['Closed PnL'] > 0).sum()
loss_trades = (filtered_df['Closed PnL'] < 0).sum()

win_rate = (
    profitable_trades /
    (profitable_trades + loss_trades)
) * 100

# ---------------- KPI CARDS ---------------- #

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Profit",
    f"${total_profit:,.0f}"
)

col2.metric(
    "Total Trades",
    f"{total_trades}"
)

col3.metric(
    "Average PnL",
    f"${avg_pnl:,.2f}"
)

col4.metric(
    "Win Rate",
    f"{win_rate:.2f}%"
)

st.divider()

# ---------------- CHARTS ---------------- #

# Profit by Sentiment
col1, col2 = st.columns(2)

with col1:
    st.subheader("Profit by Sentiment")

    profit_by_senti = (
        filtered_df
        .groupby('classification')['Closed PnL']
        .sum()
    )

    fig, ax = plt.subplots()

    profit_by_senti.plot(
        kind='bar',
        ax=ax
    )

    ax.set_ylabel("Profit")

    st.pyplot(fig)

# Average Profit by Sentiment
with col2:
    st.subheader("Average Profit by Sentiment")

    avg_profit = (
        filtered_df
        .groupby('classification')['Closed PnL']
        .mean()
    )

    fig, ax = plt.subplots()

    avg_profit.plot(
        kind='bar',
        ax=ax
    )

    ax.set_ylabel("Average PnL")

    st.pyplot(fig)

# ---------------- MONTHLY PROFIT ---------------- #

st.subheader("Monthly Profit Trend")

monthly_profit = (
    filtered_df
    .groupby('Month')['Closed PnL']
    .sum()
)

fig, ax = plt.subplots(figsize=(10, 4))

monthly_profit.plot(
    kind='line',
    marker='o',
    ax=ax
)

ax.set_ylabel("Profit")

st.pyplot(fig)

# ---------------- TOP COINS ---------------- #

st.subheader("Top Performing Coins")

coin_profit = (
    filtered_df
    .groupby('Coin')['Closed PnL']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 5))

coin_profit.plot(
    kind='bar',
    ax=ax
)

ax.set_ylabel("Profit")

st.pyplot(fig)

# ---------------- RISK ANALYSIS ---------------- #

st.subheader("Risk Analysis by Sentiment")

risk = (
    filtered_df
    .groupby('classification')['Closed PnL']
    .std()
)

st.dataframe(
    risk.reset_index().rename(
        columns={'Closed PnL': 'Risk (Std Dev)'}
    )
)

# ---------------- INSIGHTS ---------------- #

st.subheader("Key Insights")

best_sentiment = profit_by_senti.idxmax()
worst_sentiment = profit_by_senti.idxmin()

best_coin = coin_profit.idxmax()

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