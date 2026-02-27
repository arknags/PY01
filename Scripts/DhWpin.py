import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, UTC
import requests

import subprocess 
import json

# Optionally set a full path if it's stored elsewhere
CONFIG_FILE = "/users/Nags/Data/Alerts/colab/config.json"




# ========== CONFIG ==========
# Load the JSON file
with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

# Access values




# Load secrets

TELEGRAM_BOT_TOKEN = config["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = chat_id = config["TELEGRAM_CHAT_ID"]

input_file = "/users/Nags/Data/Alerts/colab/ohlcv_output_today.xlsx"  # File with all symbols

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

#============= Pin bar logic ==============
def detect_pin_bars(df):
    """
    Detects bullish and bearish pin bars in OHLCV DataFrame.

    Assumes the DataFrame has columns: ['Open', 'High', 'Low', 'Close']
    Adds a new column 'PinBar' with values: 'Bullish', 'Bearish', or None.
    """
    df1 = df.copy()  # Important: Work on a copy to avoid warnings
    # Calculate candle componentsopen'])
    df1['UpperWick'] = df1['high'] - df1[['open', 'close']].max(axis=1)
    df1['LowerWick'] = df1[['open', 'close']].min(axis=1) - df1['low']
    df1['Range'] = df1['high'] - df1['low']
    df1['Body'] = abs(df1['close'] - df1['open'])
    # Optional: Avoid divide-by-zero
    df1['Range'] = df1['Range'].replace(0, 0.001)

    # Conditions for bullish pin bar
    bullish = (
        (df1['LowerWick'] > 2 * df1['Body']) &
        (df1['UpperWick'] < df1['Body']) &
        (df1['Body'] / df1['Range'] < 0.3)
        
    )

    # Conditions for bearish pin bar
    bearish = (
        (df1['UpperWick'] > 2 * df1['Body']) &
        (df1['LowerWick'] < df1['Body']) &
        (df1['Body'] / df1['Range'] < 0.3)
    )
    print("CLOSE IS :", df1['close'], "LW :", df1['LowerWick'], "UW is", df1['UpperWick'], "Body is ",df1['Body'], "Range =", df1['Range'] )
    # Label pin bars
    df1['PinBar'] = None
    df1.loc[bullish, 'PinBar'] = 'Bullish'
    df1.loc[bearish, 'PinBar'] = 'Bearish'

    return df1

def main():

    df = pd.read_excel(input_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])

    alerts = []
    send_telegram_alert("***** 7. Weekly Pin bar scan started *********", TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    for symbol, sdf in df.groupby("symbol"):
        sdf = sdf.set_index('date')
        monthly_df = sdf.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        monthly_df['symbol'] = symbol
        monthly_df['instrument'] = sdf.iloc[0]['instrument']
    
        is_equity = (monthly_df['instrument'] == "EQUITY").all()
        if not monthly_df.empty and len(monthly_df) > 20 and is_equity:
            
            # Volume Spike Detection
            recent = monthly_df.tail(20)
            today_volume = recent.iloc[-1]["volume"]
            avg_past_12 = recent.iloc[:-1]["volume"].mean()
            vol_spike = today_volume > avg_past_12 * 2.2 # you can adjust the multiplier

            if vol_spike :        
                latest = monthly_df.iloc[-1:]
                print("Weekly data frame is " , latest)
                latest = detect_pin_bars(latest)
                pinbar_type = latest.iloc[0]['PinBar']

                if pinbar_type in ['Bullish']:
                    msg = f"📌 WEEKLY Pin Bar Detected: {pinbar_type}\n📅 {latest.iloc[0]['date'].date()}, 💹 {symbol}, 📈 Close: {latest.iloc[0]['close']}"
                    send_telegram_alert(msg, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                    alerts.append(symbol)

                # else:
                #     print("Failed Weekly df", monthly_df)
                #     print("monthly_df has insufficient rows for analysis")
                #     Msg = f"NO Weekly Pin data for {symbol}"
                #     send_telegram_alert(Msg, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    print(f"✅ Alerts sent for: {alerts}")
    if alerts:
        send_telegram_alert("***  Weekly Pin Bar Scan Completed ***", TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    else:
        send_telegram_alert("***  No Weekly Pin Bar Signals ***", TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    subprocess.run(["/Users/nags/Data/Alerts/colab/venv/bin/python", "/users/Nags/Data/Alerts/colab/DhDpin.py"])

if __name__ == "__main__":
    main()