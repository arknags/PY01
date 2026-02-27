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


def main():
    df = pd.read_excel(input_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])

    alerts = []
    send_telegram_alert("*** 9. Daily Volume check scan started ***", TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    for symbol, sdf in df.groupby(symbol_column):
        sdf = sdf.sort_values('date')
        sdf['volMean20'] = sdf['volume'].rolling(window=20).mean()
        sdf['vol_spike'] = sdf['volume'] > sdf['volMean20'] * 8

        # Check volume spike for the latest row
        if not sdf.empty and sdf.iloc[-1]['vol_spike']:
            msg = (
                f"📌 Daily Volume spike detected:\n"
                f"📅 {sdf.iloc[-1]['date'].date()}, "
                f"💹 {symbol}, "
                f"📈 Close: {sdf.iloc[-1]['close']}, "
                f"🔊 Volume: {sdf.iloc[-1]['volume']}, "
                f"📊 20 DAY Avg Volume: {round(sdf.iloc[-1]['volMean20'])}"
            )
            send_telegram_alert(msg, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
            alerts.append(symbol)
            

    print(f"✅ Alerts sent for: {alerts}")
    if alerts:
        send_telegram_alert("***  Daily Volume Scan Completed ***", TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    else:
        send_telegram_alert("***  No Daily Volume spike Bar Signals ***", TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    subprocess.run(["/Users/nags/Data/Alerts/colab/venv/bin/python", "/Users/nags/Data/Alerts/colab/DhWvo.py"])
if __name__ == "__main__":
    main()
