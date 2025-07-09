import streamlit as st
import pandas as pd
import time

# Import your bot module under a different name
# so you can pull its state. Adjust the import path as needed.
from hybrid_v2 import bst_bar, live_df, tick_df, position_open, entry_side, entry_time, cancel_all_orders

st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")

# Auto-refresh every second
st_autorefresh = st.experimental_memo(lambda: None)  # placeholder for reload trigger
st_autorefresh()

st.title("🔗 Trading Bot Live Dashboard")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    price = live_df['close'].iloc[-1] if not live_df.empty else None
    st.metric("Last Price", f"{price:.2f}" if price else "—")
with col2:
    # Hybrid score: compute it on the latest tick
    if not live_df.empty and not tick_df.empty:
        # replicate your compute_hybrid_score logic here
        st.metric("Hybrid Score", f"{(0.6*bst_bar.predict(xgb.DMatrix(live_df[bar_features].iloc[[-1]]))[0]):.3f}")
    else:
        st.metric("Hybrid Score", "—")
with col3:
    status = "Long" if position_open and entry_side==0 else "Short" if position_open else "Flat"
    st.metric("Position Status", status)
with col4:
    if st.button("🔔 Exit Position"):
        cancel_all_orders()
        st.experimental_rerun()

# Live price chart
st.subheader("Price Chart (Last 100 Bars)")
if not live_df.empty:
    st.line_chart(live_df.set_index('datetime')['close'].tail(100))

# Recent trades table (if you log them to a DataFrame)
# trades_df = pd.read_csv("trades.csv")  # or however you store them
# st.subheader("Recent Trades")
# st.dataframe(trades_df.tail(10))

st.write("⏰ Last update:", time.strftime("%Y-%m-%d %H:%M:%S"))
