import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import requests
from dotenv import load_dotenv
import subprocess 
import json

# Optionally set a full path if it's stored elsewhere
#CONFIG_FILE = "/users/Nags/Data/Alerts/colab/config.json"




# ========== CONFIG ==========
# Load the JSON file
# with open(CONFIG_FILE, 'r') as f:
#     config = json.load(f)

# Access values
bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
chat_id = os.environ.get("TELEGRAM_CHAT_ID")



# ========== CONFIG ==========
API_KEY = os.environ.get("DHAN_API_KEY")
CLIENT_ID = os.environ.get("DHAN_CHAT_ID")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(BASE_DIR, "ohlcv_output_today11.xlsx") # File with all symbols
output_file = os.path.join(BASE_DIR, "ohlcv_output_today11.xlsx") # File with all symbols
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


base_url = "https://api.dhan.co/v2"

IST = timezone(timedelta(hours=5, minutes=30))


headers = {
    "access-token": API_KEY,
    "Content-Type": "application/json",
    "accept": "application/json"
}
headers_with_client = {
    **headers,
    "client-id": CLIENT_ID
}

def get_today_ohlcv(security_id, eS):
    url = f"{base_url}/marketfeed/quote"
    time.sleep(1)
    if eS == "IDX_I":
        r = requests.post(url, headers=headers_with_client, json={"IDX_I": [security_id]})
        #print (r.json().get("data", {}).get("IDX_I", {}).get(str(security_id), {}))
        return r.json().get("data", {}).get("IDX_I", {}).get(str(security_id), {})
    else:
       r = requests.post(url, headers=headers_with_client, json={"NSE_EQ": [security_id]})
       return r.json().get("data", {}).get("NSE_EQ", {}).get(str(security_id), {})
   

# Build OHLCV DataFrame
def fetch_ohlcv_dhan(instrument, security_id, fromDate, toDate):
 
    # Define eS within the scope of fetch_ohlcv_dhan
    eS = "NSE_EQ" if instrument == "EQUITY" else "IDX_I"
    from_date = fromDate
    to_date = toDate
    df = pd.DataFrame()

    # Step# 2 Skip fetching if today is Saturday or Sunday
    toDate1 = datetime.strptime(toDate, "%Y-%m-%d")
    weekday = toDate1.strftime('%A')
    if weekday in ['Saturday','Sunday']:
        print(f"Today is {weekday} ({toDate1}), skipping today's OHLC/close/volume fetch.")
        today_data = None
    else:
        today_ohlcv = get_today_ohlcv(security_id, eS)


        if today_ohlcv and 'ohlc' in today_ohlcv :
            today_data = {
                "date": toDate1,
                "open": today_ohlcv["ohlc"].get("open"),
                "high": today_ohlcv["ohlc"].get("high"),
                "low": today_ohlcv["ohlc"].get("low"),
                "close": today_ohlcv["last_price"],
                "volume": today_ohlcv.get("volume") if eS == "NSE_EQ" else 321321321
            }

        else :
            return None

    if not df.empty:
        # Convert date column to datetime objects before sorting if they aren't already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').reset_index(drop=True)
    # Check volSpike
    df = pd.DataFrame([today_data])
    return df

# ---- User Configuration ----
excel_path = os.path.join(BASE_DIR, "stockfo1.xlsx")
input_path = os.path.join(BASE_DIR, "ohlcv_output_today11.xlsx")
output_file = os.path.join(BASE_DIR, "ohlcv_output_today11.xlsx")
sheet_name = "Sheet1"
fromDate = "2024-01-01"
#toDate = "2025-05-28"
toDate = datetime.now().strftime("%Y-%m-%d")

# ---- Excel Loading ----
try:

    excel_df = pd.read_excel(excel_path, sheet_name=sheet_name)
    print(f"✅ Loaded {len(excel_df)} rows from Excel.")
except Exception as e:
    print(f"❌ Failed to read Excel: {e}")
    excel_df = pd.DataFrame()

# ---- Validate Columns ----
if not {"securityId", "instrument"}.issubset(excel_df.columns):
    print("❌ Excel must contain 'securityId' and 'instrument' columns.")
else:
    all_dataframes = []
    df1 = pd.read_excel(input_path)
    all_dataframes.append(df1)
    #print("Before append", all_dataframes)
    toDate1 = datetime.strptime(toDate, "%Y-%m-%d")
    weekday = toDate1.strftime('%A')
    if weekday in ['Saturday', 'Sunday']:
        print(f"Today is {weekday} ({toDate1}), skipping today's OHLC/close/volume fetch.")
        msg = f"Weekend, no OHLCV fetched for today ({toDate})\n"
        send_telegram_alert(msg, bot_token, chat_id)
        #subprocess.run(["/Users/nags/Data/Alerts/colab/venv/bin/python", "/Users/nags/Data/Alerts/colab/DhMtest.py"])
    else:
        msg = f"Daily OHLCV fetch for today ({toDate}) in progress\n"
        send_telegram_alert(msg, bot_token, chat_id)
        L1 = 0
        for idx, row in excel_df.iterrows():
            try:
                L1 = L1 + 1
                instrument = row["instrument"]
                symbol = row["symbol"]
                securityId = int(row["securityId"])
                print(f"\nFetching: {L1} - Symbol = {symbol} , Instrument={instrument}, ID={securityId}")
                df = fetch_ohlcv_dhan(instrument, securityId, fromDate, toDate)
                # --- NEW API VALIDATION CHECK ---
                if L1 == 1 :
                    # If the first attempt returns None or an empty DF, or the 'open' price is missing
                    if df is None or df.empty or df.iloc[0]['open'] is None:
                        error_msg = f"🛑 CRITICAL: API Validation Failed at {symbol}. Likely invalid API Key or Session. Exiting script."
                        print(error_msg)
                        send_telegram_alert(error_msg, bot_token, chat_id)
                        import sys
                        sys.exit()
                # --------------------------------
                if df is not None and not df.empty:
                    df["instrument"] = instrument
                    df["securityId"] = securityId
                    df["symbol"] = symbol
                    print("Today data", df)
                    all_dataframes.append(df)
                    #print("After append", all_dataframes)
                else:
                    print(f"⚠️ No data fetched for Symbol = {symbol}")
                if L1 in [100, 200, 300, 400, 500, 600, 700]:
                    send_telegram_alert(f"{L1} fetch completed", bot_token, chat_id)

            except Exception as err:
                print(f"⚠️ Error for NK1row {idx}: {err}")

    # ---- Save Outputs ----
        if all_dataframes:
            final_df = pd.concat(all_dataframes, ignore_index=True)
            final_df.to_excel(output_file, index=False)
            print("✅ OHLCV data saved to /content/ohlcv_output_today.xlsx")
        else:
            print("❌ No daily OHLCV data fetched.")


        msg = f"Fetched OHLCV successfully for today ({toDate})\n"
        send_telegram_alert(msg, bot_token, chat_id)

