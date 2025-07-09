import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from collections import deque
import time
import threading
from signalrcore.hub_connection_builder import HubConnectionBuilder
from datetime import datetime, timedelta, timezone
from sklearn.metrics import accuracy_score, log_loss
import math

# === Periodic Save ===
SAVE_INTERVAL_MIN = 60

def periodic_save(interval_min=SAVE_INTERVAL_MIN):
    while True:
        time.sleep(interval_min * 60)
        ts = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')
        live_df.to_csv(f"live_bars_{ts}.csv", index=False)
        tick_df.to_csv(f"tick_data_{ts}.csv", index=False)
        print(f"💾 Auto-saved data at {ts}")

# === Configuration ===
API_BASE_URL = "https://api.topstepx.com"
USERNAME      = "bhart012"
API_KEY       = "fqTIW2rYLtem35WBT7CahWD9XMFT0Q+gr9hNET5cxWM="
MODEL_PATH    = "trained_model.bin"

# === Parameters & Globals ===
tick_size             = 0.25

TICK_WINDOW           = 50          # number of ticks to buffer for features
BAR_RETRAIN_INTERVAL  = 150         # retrain bar model every ~2.5 hours
THRESHOLD_WINDOW      = 200         # history size for dynamic thresholds
UP_STD_MULT           = 1.0
DOWN_STD_MULT         = 1.0

MIN_TRADE_DURATION_SEC = 120        # don’t reversal‐exit until 2 min after entry
COOLDOWN_SEC           = 60         # wait 60s after exit before re‐entry
REVERSAL_DELTA         = 0.15       # require 15% flip to reverse

# ATR multipliers for bracket orders
TP_ATR_MULT            = 2.0        # take‐profit at 2×ATR
SL_ATR_MULT            = 1.0        # stop‐loss at 1×ATR

# Absolute minimum width (in ticks) to avoid one-tick stopouts
MIN_TICKS              = 4

# Tick‐model retrain cadence
TICK_RETRAIN_INTERVAL  = 500        # ticks before retrain tick model

# History for dynamic thresholds
hybrid_history = deque(maxlen=THRESHOLD_WINDOW)

# === Global State ===
token        = None
account_id   = None
contract_id  = None
position_open = False
bar_counter  = 0
tick_counter = 0
bst_bar      = None
bst_tick     = None
retrain_lock = threading.Lock()
entry_side   = None
entry_score  = None
entry_time   = None
last_exit_time = None

# Typed DataFrames
data_types = {
    'datetime': 'datetime64[ns]', 'open': float, 'high': float,
    'low': 'float64', 'close': float, 'volume': float
}
live_df = pd.DataFrame({k: pd.Series(dtype=v) for k, v in data_types.items()})
tick_df = pd.DataFrame({
    'timestamp': pd.Series(dtype='datetime64[ns]'),
    'price': pd.Series(dtype=float),
    'volume': pd.Series(dtype=float)
})

bar_features        = ['returns','ma_21','ma_50','atr_14','volatility_regime','rsi_14','bollinger_width']
tick_feature_names  = ['tick_return','tick_momentum','tick_volatility','order_flow_imbalance','tick_vwap']

# === Utilities ===
def round_to_tick(p):
    return round(p / tick_size) * tick_size

def _exit_position():
    global position_open, entry_side, entry_score
    print("🔔 Reversal exit, flattening...")
    cancel_all_orders()
    flat = 1 - entry_side
    resp = requests.post(
        f"{API_BASE_URL}/api/Order/place",
        json={"accountId": account_id, "contractId": contract_id, "type": 2, "side": flat, "size": 1},
        headers={"Authorization": f"Bearer {token}"}
    )
    resp.raise_for_status()
    position_open = False
    entry_side    = None
    entry_score   = None

# === API/Auth ===
def get_access_token():
    r = requests.post(
        f"{API_BASE_URL}/api/Auth/loginKey",
        json={"userName": USERNAME, "apiKey": API_KEY}
    )
    r.raise_for_status()
    return r.json()["token"]

def search_accounts():
    r = requests.post(
        f"{API_BASE_URL}/api/Account/search",
        json={"onlyActiveAccounts": True},
        headers={"Authorization": f"Bearer {token}"}
    )
    r.raise_for_status()
    return r.json().get("accounts", [])

def search_nq_contracts():
    r = requests.post(
        f"{API_BASE_URL}/api/Contract/search",
        json={"live": False, "searchText": "NQ"},
        headers={"Authorization": f"Bearer {token}"}
    )
    r.raise_for_status()
    ctr = r.json().get("contracts", [])
    return sorted([c for c in ctr if "NQ" in c['id'] and c.get('activeContract', False)], key=lambda x: x['id'], reverse=True)

# === Indicators ===
def update_bar_indicators(df):
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['returns']          = df['close'].pct_change()
    df['ma_21']            = df['close'].rolling(21).mean()
    df['ma_50']            = df['close'].rolling(50).mean()
    tr = df.apply(lambda r: max(r['high']-r['low'], abs(r['high']-r['close']), abs(r['low']-r['close'])), axis=1)
    df['atr_14']           = tr.rolling(14).mean()
    df['atr_50']           = tr.rolling(50).mean()
    df['volatility_regime'] = (df['atr_14'] > df['atr_50']).astype(int)
    delta = df['close'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi_14']          = 100 - (100/(1+rs))
    mid = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bollinger_width'] = (2*std)/(mid+1e-9)

# === Bar Model Training ===
def retrain_bar_model(df):
    global bst_bar
    if any(f not in df.columns for f in bar_features):
        print("⚠️ Waiting for indicators")
        return
    def _train():
        if retrain_lock.locked(): return
        with retrain_lock:
            c = df.dropna(subset=bar_features)
            if len(c) < BAR_RETRAIN_INTERVAL: return
            X = c[bar_features][:-1]
            y = (c['close'].shift(-1) > c['close']).astype(int)[:-1]
            m = xgb.train({"objective":"binary:logistic","eval_metric":"logloss"},
                          xgb.DMatrix(X, label=y), num_boost_round=75)
            preds = m.predict(xgb.DMatrix(X))
            print(f"Bar Acc:{accuracy_score(y,preds>0.5):.4f}, LL:{log_loss(y,preds):.4f}")
            bst_bar = m
            m.save_model(MODEL_PATH)
    threading.Thread(target=_train, daemon=True).start()

# === Tick Model Training ===
def retrain_tick_model(df):
    global bst_tick
    if len(df) < TICK_RETRAIN_INTERVAL: return
    feats = []; labels = []
    for i in range(TICK_WINDOW, len(df)-1):
        f = compute_tick_features_at_index(df, i)
        feats.append([f[n] for n in tick_feature_names])
        labels.append(int(df['price'].iloc[i+1] > df['price'].iloc[i]))
    D = xgb.DMatrix(np.array(feats), label=np.array(labels), feature_names=tick_feature_names)
    bst_tick = xgb.train({"objective":"binary:logistic","eval_metric":"logloss"}, D, num_boost_round=50)
    print("🔄 ✅ Tick model retrained")

def compute_tick_features_at_index(df, idx):
    # wrapper that resets index
    window = df.iloc[idx-TICK_WINDOW:idx].copy().reset_index(drop=True)
    return compute_tick_features(n=TICK_WINDOW)

# === Historical Data ===
def load_historical_data():
    now = datetime.now(timezone.utc)
    st  = now - timedelta(days=30)
    r = requests.post(
        f"{API_BASE_URL}/api/History/retrieveBars",
        json={
            "contractId": contract_id,
            "live": False,
            "startTime": st.isoformat(),
            "endTime": now.isoformat(),
            "unit": 2,
            "unitNumber": 1,
            "limit": 5000
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    r.raise_for_status()
    bars = r.json().get("bars", [])
    rec = [{
        'datetime': pd.to_datetime(b['t'], utc=True),
        'open': b['o'], 'high': b['h'],
        'low': b['l'], 'close': b['c'], 'volume': b['v']
    } for b in bars]
    df = pd.DataFrame(rec)
    update_bar_indicators(df)
    global live_df
    live_df = df.copy()
    retrain_bar_model(live_df)

# === Tick Features ===
def compute_tick_features(n=TICK_WINDOW):
    window = tick_df.tail(n).copy().reset_index(drop=True)
    if len(window) < n:
        return {k: np.nan for k in tick_feature_names}
    window.loc[:, 'tick_return']     = window['price'].pct_change()
    window.loc[:, 'tick_momentum']   = window['price'] - window['price'].iloc[0]
    window.loc[:, 'tick_volatility'] = window['price'].rolling(n).std()
    window.loc[:, 'up']              = (window['price'] > window['price'].shift(1)).astype(int)
    window.loc[:, 'dn']              = (window['price'] < window['price'].shift(1)).astype(int)
    ofi = (window['up'] - window['dn']).sum()
    cum_vol  = window['volume'].cumsum() + 1e-9
    cum_vwap = (window['price'] * window['volume']).cumsum() / cum_vol
    last = window.iloc[-1]
    return {
        'tick_return': last['tick_return'],
        'tick_momentum': last['tick_momentum'],
        'tick_volatility': last['tick_volatility'],
        'order_flow_imbalance': ofi,
        'tick_vwap': cum_vwap.iloc[-1]
    }

# === Order Management ===
def cancel_all_orders():
    r = requests.post(
        f"{API_BASE_URL}/api/Order/searchOpen",
        json={"accountId": account_id},
        headers={"Authorization": f"Bearer {token}"}
    )
    r.raise_for_status()
    orders = r.json().get("orders", [])
    for o in orders:
        requests.post(
            f"{API_BASE_URL}/api/Order/cancel",
            json={"accountId": account_id, "orderId": o['id']},
            headers={"Authorization": f"Bearer {token}"}
        )
    print(f"✅ Canceled {len(orders)} open orders.")

# === Positions ===
def search_open_positions():
    r = requests.post(
        f"{API_BASE_URL}/api/Position/searchOpen",
        json={"accountId": account_id},
        headers={"Authorization": f"Bearer {token}"}
    )
    r.raise_for_status()
    return len(r.json().get("positions", [])) > 0

def position_sync_loop():
    global position_open
    while True:
        p = search_open_positions()
        if p != position_open:
            position_open = p
            if not p:
                cancel_all_orders()
        time.sleep(5)

# === Live Feed & Hybrid Logic ===
def on_quote(args):
    global tick_counter, live_df, tick_df, bar_counter
    global position_open, entry_side, entry_score, entry_time, last_exit_time

    # 1) Debug receipt
    q     = args[1]
    ts    = pd.to_datetime(q['timestamp'], utc=True)
    price = q['lastPrice']
    #print("🔔 Received tick:", price, "at", ts.time())

    # 2) Tick buffer
    tick_df = pd.concat([
        tick_df,
        pd.DataFrame([{'timestamp': ts, 'price': price, 'volume': q.get('volume', 0)}])
    ], ignore_index=True)
    if len(tick_df) > 500:
        tick_df = tick_df.tail(500).reset_index(drop=True)

    # 3) Count ticks
    tick_counter += 1
    #print(f"🔢 Tick buffer: {tick_counter}/{TICK_WINDOW}")

    # 4) Wait until buffer filled
    if tick_counter < TICK_WINDOW:
        print(f"⏳ Accumulating ticks: {tick_counter}/{TICK_WINDOW}")
        return

    # 5) Update bars on new minute
    bt = ts.floor('min')
    if live_df.empty or live_df.iloc[-1]['datetime'] != bt:
        live_df = pd.concat([
            live_df,
            pd.DataFrame([{
                'datetime': bt, 'open': price,
                'high': price,  'low': price,
                'close': price, 'volume': q.get('volume', 0)
            }])
        ], ignore_index=True)
        update_bar_indicators(live_df)

        bar_counter += 1
        if bar_counter >= BAR_RETRAIN_INTERVAL:
            retrain_bar_model(live_df)
            bar_counter = 0

    latest = live_df.iloc[-1]
    if np.isnan(latest['atr_14']):
        return

    # 6) Compute hybrid score
    prob_bar    = bst_bar.predict(
        xgb.DMatrix(latest[bar_features].values.reshape(1, -1),
                    feature_names=bar_features)
    )[0]
    feats        = compute_tick_features(TICK_WINDOW)
    mom_sig      = 1 if feats['tick_momentum'] > 0 else 0
    ofi_sig      = 1 if feats['order_flow_imbalance'] > 0 else 0
    vwap_sig     = 1 if price > feats['tick_vwap'] else 0
    hybrid_score = 0.6*prob_bar + 0.2*mom_sig + 0.1*ofi_sig + 0.1*vwap_sig
    print(f"📊 Hybrid score: {hybrid_score:.3f} (Bar:{prob_bar:.3f}, Mom:{mom_sig}, OFI:{ofi_sig}, VWAP:{vwap_sig})")

    # 7) Dynamic thresholds
    hybrid_history.append(hybrid_score)
    if len(hybrid_history) >= THRESHOLD_WINDOW:
        mu    = np.mean(hybrid_history)
        sigma = np.std(hybrid_history)
        up_thresh   = mu + UP_STD_MULT   * sigma
        down_thresh = mu - DOWN_STD_MULT * sigma
    else:
        up_thresh, down_thresh = 0.6, 0.4
    print(f"🔧 Thresholds → Long > {up_thresh:.3f}, Short < {down_thresh:.3f}")

    now_ts = time.time()

    # 8) Reversal‐exit
    if position_open and entry_time is not None:
        if now_ts - entry_time >= MIN_TRADE_DURATION_SEC:
            if entry_side == 0:
                swing_exit = entry_score - REVERSAL_DELTA
                if hybrid_score < down_thresh and hybrid_score < swing_exit:
                    print(f"🔻 Reversal exit long @ {hybrid_score:.3f}")
                    _exit_position(); last_exit_time = now_ts; return
            else:
                swing_exit = entry_score + REVERSAL_DELTA
                if hybrid_score > up_thresh and hybrid_score > swing_exit:
                    print(f"🚀 Reversal exit short @ {hybrid_score:.3f}")
                    _exit_position(); last_exit_time = now_ts; return

    # 9) Entry logic with cooldown & bar‐model filter
    can_enter = (not position_open) and (
        last_exit_time is None or now_ts - last_exit_time >= COOLDOWN_SEC
    )
    if can_enter and prob_bar > 0.5:
        if hybrid_score > up_thresh:
            entry_side  = 0
            entry_score = hybrid_score
            entry_time  = now_ts
            print("🚀 Hybrid Long Signal")
            place_bracket_order(
                side=0, size=1,
                entry_price=price,
                tp_atr_mult=TP_ATR_MULT,
                sl_atr_mult=SL_ATR_MULT
            )
        elif hybrid_score < down_thresh:
            entry_side  = 1
            entry_score = hybrid_score
            entry_time  = now_ts
            print("🔻 Hybrid Short Signal")
            place_bracket_order(
                side=1, size=1,
                entry_price=price,
                tp_atr_mult=TP_ATR_MULT,
                sl_atr_mult=SL_ATR_MULT
            )

# === Market Feed with Reconnect ===
def start_market_feed():
    while True:
        try:
            hub = HubConnectionBuilder()\
                .with_url(f"https://rtc.topstepx.com/hubs/market?access_token={token}")\
                .build()
            alive = {'flag': True}
            hub.on_open(lambda: print("🔓 Connection opened"))
            hub.on_close(lambda: alive.update(flag=False) or print("❌ Connection closed, reconnecting..."))
            hub.on("GatewayQuote", on_quote)
            hub.start()
            time.sleep(1)
            hub.send("SubscribeContractQuotes", [contract_id])
            print("🚀 Market Feed Started and subscribed.")
            while alive['flag']:
                time.sleep(1)
        except Exception as e:
            print(f"❌ Market feed error: {e}, retrying in 5s...")
            time.sleep(5)

# === Bracket Order ===
def place_bracket_order(side, size, entry_price, tp_atr_mult, sl_atr_mult):
    global position_open, entry_side, entry_score

    latest_atr = live_df['atr_14'].iloc[-1]

    # compute raw SL/TP based on ATR
    tp_raw = entry_price + tp_atr_mult * latest_atr
    sl_raw = entry_price - sl_atr_mult * latest_atr

    # enforce absolute minimum width
    min_width = MIN_TICKS * tick_size
    if abs(tp_raw - entry_price) < min_width:
        tp_raw = entry_price + math.copysign(min_width, tp_raw - entry_price)
    if abs(entry_price - sl_raw) < min_width:
        sl_raw = entry_price - math.copysign(min_width, entry_price - sl_raw)

    # round to tick
    e  = round_to_tick(entry_price)
    tp = round_to_tick(tp_raw)
    sl = round_to_tick(sl_raw)

    print(f"🧮 DEBUG → Entry:{e}, TP:{tp}, SL:{sl} "
          f"(ATR×{tp_atr_mult}/{sl_atr_mult}, min_ticks={MIN_TICKS})")

    headers = {"Authorization": f"Bearer {token}"}

    # Entry leg
    resp = requests.post(
        f"{API_BASE_URL}/api/Order/place",
        json={"accountId": account_id, "contractId": contract_id,
              "type": 2, "side": side, "size": size, "limitPrice": e},
        headers=headers
    )
    resp.raise_for_status()
    oid = resp.json().get("orderId")
    print(f"✅ Entry placed → ID: {oid}")

    # Stop-loss leg
    resp = requests.post(
        f"{API_BASE_URL}/api/Order/place",
        json={"accountId": account_id, "contractId": contract_id,
              "type": 4, "side": 1 - side, "size": size,
              "stopPrice": sl, "linkedOrderId": oid},
        headers=headers
    )
    resp.raise_for_status()
    print(f"🛡 Stop-loss placed → {sl}")

    # Take-profit leg
    resp = requests.post(
        f"{API_BASE_URL}/api/Order/place",
        json={"accountId": account_id, "contractId": contract_id,
              "type": 1, "side": 1 - side, "size": size,
              "limitPrice": tp, "linkedOrderId": oid},
        headers=headers
    )
    resp.raise_for_status()
    print(f"🎯 Take-profit placed → {tp}")

    position_open = True
    print(f"🚀 Bracket done (Entry ID: {oid})\n")

# === Main Execution ===
if __name__ == "__main__":
    token = get_access_token()
    print("Available Accounts:")
    accounts = search_accounts()
    for i, a in enumerate(accounts):
        print(f"[{i}] {a['name']} (ID: {a['id']})")
    account_id = accounts[int(input("Select account #: "))]['id']

    print("Available NQ Contracts:")
    contracts = search_nq_contracts()
    for i, c in enumerate(contracts):
        print(f"[{i}] {c['id']}")
    contract_id = contracts[int(input("Select contract #: "))]['id']

    try:
        bst_bar = xgb.Booster()
        bst_bar.load_model(MODEL_PATH)
        print("✅ Bar model loaded from disk.")
    except Exception:
        print("⚠️ No existing model found, will retrain on historical data.")

    threading.Thread(target=periodic_save, daemon=True).start()
    load_historical_data()

    position_open = search_open_positions()
    threading.Thread(target=position_sync_loop, daemon=True).start()
    threading.Thread(target=start_market_feed, daemon=True).start()

    print("Bot running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Interrupted. Exporting data for backtest...")
        live_df.to_csv(r"C:\Users\bhart\Downloads\live_bars.csv", index=False)
        tick_df.to_csv(r"C:\Users\bhart\Downloads\tick_data.csv", index=False)
        print("✅ Saved live_bars.csv and tick_data.csv")
        exit()
