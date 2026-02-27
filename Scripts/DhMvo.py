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


import pandas as pd

def main():


    df = pd.read_excel(input_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values([symbol_column, 'date'])

    alerts = []
    send_telegram_alert("*** 11. Monthly Volume check scan started ***", TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    for symbol, sdf in df.groupby(symbol_column):
        sdf = sdf.set_index('date')
        monthly_df = sdf.resample('ME').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        monthly_df['symbol'] = symbol
        monthly_df['instrument'] = sdf.iloc[0]['instrument']

        if not monthly_df.empty and len(monthly_df) > 13:
            recent = monthly_df.tail(13)
            today_volume = recent.iloc[-1]["volume"]
            avg_past_12 = recent["volume"].tail(13).mean()

            vol_spike = today_volume > avg_past_12  * 5  # You can apply a multiplier like avg_past_12 * 2

            if vol_spike:
                msg = (
                    f"📌 Monthly Volume spike detected:\n"
                    f"📅 {recent.iloc[-1]['date'].date()}, "
                    f"💹 {symbol}, "
                    f"📈 Close: {recent.iloc[-1]['close']}, "
                    f"🔊 Volume: {today_volume}, "
                    f"📊 12M Avg Volume: {round(avg_past_12)}"
                )
                send_telegram_alert(msg, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                alerts.append(symbol)


    if alerts:
        send_telegram_alert("***  Monthly Volume Scan Completed *****", TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    else:
        send_telegram_alert("***  No Monthly Volume spike Bar Signals ***", TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    send_telegram_alert("*****  ALL SCRIPT EXECUTED *****", TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    #subprocess.run(["/Users/nags/Data/Alerts/colab/venv/bin/python", "/Users/nags/Data/Alerts/colab/DhMvo.py"])
if __name__ == "__main__":
    main()
