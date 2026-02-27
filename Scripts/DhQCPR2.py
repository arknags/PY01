import pandas as pd
import numpy as np
import requests
import subprocess 
from datetime import datetime
import json

# Optionally set a full path if it's stored elsewhere
CONFIG_FILE = "/users/Nags/Data/Alerts/colab/config.json"




# ========== CONFIG ==========
# Load the JSON file
with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

# Access values
bot_token = config["TELEGRAM_BOT_TOKEN"]
chat_id = config["TELEGRAM_CHAT_ID"]


# ========== CONFIG ==========
input_file = "/users/Nags/Data/Alerts/colab/ohlcv_output_today.xlsx"
symbol_column = "symbol"


# ========== TELEGRAM FUNCTION ==========
def send_telegram_alert(message, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=payload)
    if response.status_code != 200:
        print(f"❌ Telegram Error: {response.text}")
    else:
        print(f"📩 Alert sent: {message}")

# ========== CPR CALCULATION ==========
def calculate_cpr_levels(df):
    df = df.copy()
    df["bc"] = (df["high"] + df["low"]) / 2
    df["pivot"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tc"] = 2 * df["pivot"] - df["bc"]

    # Ensure tc above pivot and bc below
    df["tc"], df["bc"] = df[["tc", "bc"]].max(axis=1), df[["tc", "bc"]].min(axis=1)

    df["r1"] = 2 * df["pivot"] - df["low"]
    df["s1"] = 2 * df["pivot"] - df["high"]
    df["r2"] = df["pivot"] + (df["r1"] - df["s1"])
    df["r3"] = df["high"] + 2 * (df["pivot"] - df["low"])
    df["r4"] = df["high"] + 3 * (df["pivot"] - df["low"])
    df["s2"] = df["pivot"] - (df["r1"] - df["s1"])
    df["s3"] = df["low"] - 2 * (df["high"] - df["pivot"])
    df["s4"] = df["low"] - 3 * (df["high"] - df["pivot"])
    return df

# ========== CPR SIGNALS ==========
def check_cpr_signals(row, level_name, proximity_pct=0.15):
    level = row[level_name]
    if pd.isna(level): return None

    open_, high, low, close = row['open'], row['high'], row['low'], row['close']
    body = abs(close - open_)
    range1 = abs(low - high)
    lower_wick = min(open_, close) - low
    upper_wick = high - max(open_, close)

    bproximity = low > level and abs(low - level) / level * 100 <= proximity_pct
    sproximity = high < level and abs(high - level) / level * 100 <= proximity_pct
    btouch = low <= level <= high and close > level and low < level
    stouch = high >= level >= low and close < level and high > level

    bbounce = (bproximity or btouch) and (close > level) and ((lower_wick > 2 * body and upper_wick < body and body / range1 < 0.5)  or ((body > 0.75 * range1) and (high - level) > 0.5 * range1) and (close > open_)  )

    srejection = (sproximity or stouch) and (close < level) and ( (upper_wick > 2 * body and lower_wick < body and body / range1 < 0.5)   or ((body > 0.75 * range1) and (level - low) > 0.5 * range1)  and (close < open_) )

    return {
        f"{level_name}_BUY touch": btouch,
        #f"{level_name}_SELL touch": stouch
        # f"{level_name}_BUY bounce": bbounce,
        # f"{level_name}_BUY proximity": bproximity,
        #f"{level_name}_SELL proximity": sproximity,
        #f"{level_name}_SELL rejection": srejection
        
    }

# ========== MAIN ==========
def main():
    df = pd.read_excel(input_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values([symbol_column, 'date'])

    alerts = []
    all_final_signals = []
    send_telegram_alert("---5. QCPR started", bot_token, chat_id)
    for symbol, sdf in df.groupby(symbol_column):
        sdf = sdf.set_index('date')
        monthly_df = sdf.resample('QE').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        monthly_df['symbol'] = symbol

        if len(monthly_df) < 2:
            print(f"⚠️ Skipping {symbol}: Only {len(monthly_df)} quarterly rows after resample.")
            continue

        monthly_df['volMean20'] = monthly_df['volume'].rolling(window=4).mean()
        monthly_df['volSpike'] = monthly_df['volume'] > monthly_df['volMean20']
        L1 = monthly_df.iloc[-1]
        
        if not L1['volSpike']:
            continue
        #print(f"{symbol} | Volume: {L1['volume']} | Mean20: {L1['volMean20']} | volSpike: {L1['volSpike']}")
        monthly_cpr_df = calculate_cpr_levels(monthly_df)
        monthly_cpr_df = monthly_cpr_df.sort_values("date").reset_index(drop=True)

        for col in ["pivot", "tc", "bc", "r1", "r2", "r3", "r4", "s1", "s2", "s3", "s4"]:
            monthly_cpr_df[col] = monthly_cpr_df[col].shift(1)

        monthly_cpr_df.iloc[0, monthly_cpr_df.columns.get_indexer(["pivot", "tc", "bc", "r1", "r2", "r3", "r4", "s1", "s2", "s3", "s4"])] = 1
        monthly_cpr_df["month"] = monthly_cpr_df["date"].dt.to_period("Q")
        sdf["month"] = sdf.index.to_period("Q")

        merged_df = sdf.merge(
            monthly_cpr_df[["month", "pivot", "tc", "bc", "r1", "r2", "r3", "r4", "s1", "s2", "s3", "s4"]],
            on="month", how="left"
        )

        last_row = merged_df.iloc[-1]
        print("Last QCPR is : ", last_row)
        cpr_levels = ['pivot', 'tc', 'bc', 'r1', 'r2', 's1', 's2']

        signals = {
            k: v for level in cpr_levels
            for k, v in (check_cpr_signals(last_row, level) or {}).items()
        }

        signal_df = pd.DataFrame([signals])
        if signal_df.iloc[0].eq(False).all():
            print(f"❌ No Quarterly CPR Signals for {symbol}")
            continue

        final_signals = pd.concat([merged_df.tail(1).reset_index(drop=True), signal_df], axis=1)
        final_signals["symbol"] = symbol
        all_final_signals.append(final_signals)

        true_signals = [col for col, val in signal_df.iloc[0].items() if val]
        alert_message = f"🚨 {datetime.now().date()} Quarterly CPR Signal for {symbol}, close = {last_row['close']:.2f}:\n"
        alert_message += "\n".join([f"{lvl.upper()} = {last_row[lvl]:.2f}" for lvl in cpr_levels])
        alert_message += "\n".join([f"✅ {sig}" for sig in true_signals])
        send_telegram_alert(alert_message, bot_token, chat_id)

    # Save final signals if any
    if all_final_signals:
        final_df = pd.concat(all_final_signals, ignore_index=True)
        final_df.to_excel("/users/nags/Data/Alerts/colab/final_monthly_signals.xlsx", index=False)
        print("✅ Alerts saved to final_quarterly_signals.xlsx")
        send_telegram_alert("-----QCPR check Completed-----", bot_token, chat_id)
    else:
        print("❌ No QCPR signals generated for any symbol")
        send_telegram_alert("NO QCPR signals", bot_token, chat_id)

    subprocess.run(["/Users/nags/Data/Alerts/colab/venv/bin/python", "/Users/nags/Data/Alerts/colab/DhMpin.py"])
if __name__ == "__main__":
    main()
