import pandas as pd
import numpy as np
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
bot_token = config["TELEGRAM_BOT_TOKEN"]
chat_id = config["TELEGRAM_CHAT_ID"]

# ========== CONFIG ==========
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

# ========== SUPERTREND ==========
def compute_supertrend(df, period=2, multiplier=3):
    high, low, close = df['high'], df['low'], df['close']
    tr1, tr2, tr3 = high - low, (high - close.shift()).abs(), (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    hl2 = (high + low) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    supertrend = np.full(len(df), np.nan)
    trend = np.full(len(df), True, dtype=bool)

    if close.iloc[1] > upperband.iloc[1]:
        trend[0], supertrend[0] = True, lowerband.iloc[0]
    elif close.iloc[1] < lowerband.iloc[1]:
        trend[0], supertrend[0] = False, upperband.iloc[0]
    else:
        trend[0] = close.iloc[0] >= close.iloc[1]
        supertrend[0] = lowerband.iloc[0] if trend[0] else upperband.iloc[0]

    for i in range(1, len(df)):
        curr_close = close.iloc[i]
        prev_supertrend = supertrend[i - 1]
        prev_trend = trend[i - 1]

        curr_upperband, curr_lowerband = upperband.iloc[i], lowerband.iloc[i]
        if prev_trend and curr_lowerband < prev_supertrend:
            curr_lowerband = prev_supertrend
        if not prev_trend and curr_upperband > prev_supertrend:
            curr_upperband = prev_supertrend

        if prev_trend:
            if curr_close < curr_lowerband:
                trend[i], supertrend[i] = False, curr_upperband
            else:
                trend[i], supertrend[i] = True, curr_lowerband
        else:
            if curr_close > curr_upperband:
                trend[i], supertrend[i] = True, curr_lowerband
            else:
                trend[i], supertrend[i] = False, curr_upperband

    df['ATR'] = atr
    df['UpperBand'] = upperband
    df['LowerBand'] = lowerband
    df['Supertrend'] = supertrend
    df['Trend'] = np.where(trend, 'Uptrend', 'Downtrend')
    return df

# ========== SIGNAL CHECK ==========
def check_supertrend_signals(row, level_name='Supertrend', proximity_pct=30):
    level = row.get(level_name)
    atr = row.get('ATR')
    if pd.isna(level) or pd.isna(atr): return None

    proximity = atr * proximity_pct / 100
    open_, high, low, close = row['open'], row['high'], row['low'], row['close']
    body = abs(close - open_)
    lower_wick = min(open_, close) - low
    upper_wick = high - max(open_, close)
    range1 = abs(high - low)

    bproximity = low > level and low < (level + proximity)
    sproximity = high < level and high > (level - proximity)
    btouch = (low <= level <= high) and close > level  
    stouch = high >= level >= low

    bbounce = (bproximity or btouch) and (close > level) and (
        (lower_wick > 2 * body and upper_wick < body and body / range1 < 0.5) or
        ((body > 0.5 * range1) or (high - level) > 0.5 * range1)
    )

    srejection = (sproximity or stouch) and (close < level) and (
        (upper_wick > 2 * body and lower_wick < body and body / range1 < 0.5) or
        ((body > 0.5 * range1) or (level - low) > 0.5 * range1)
    )

    return {
        "WST_BUY TOUCH" :btouch,
        "WST_BUY bounce": bbounce,
        "WST_BUY proximity": bproximity,
        "WST_SELL rejection": srejection,  # optional
        "WST_SELL proximity": sproximity
    }

# ========== MAIN ==========
def main():
    df = pd.read_excel(input_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])

    alerts = []
    send_telegram_alert("***** 2. WST started *******", bot_token, chat_id)
    for symbol, sdf in df.groupby(symbol_column):
        sdf = sdf.set_index('date')
        weekly_df = sdf.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        weekly_df['symbol'] = symbol
     
        #print("Weekly DF is", weekly_df.tail(1))
        if len(weekly_df) >= 2:
            weekly_df = compute_supertrend(weekly_df)
        

            latest = weekly_df.iloc[-1]
            prevweek = weekly_df.iloc[-2]
            signals = check_supertrend_signals(latest)
            #print("Weekly DF is :" , weekly_df)
            #if signals and (signals["WST_BUY bounce"] or signals["WST_BUY proximity"]):
            #if (signals["WST_BUY proximity"] or signals["WST_BUY TOUCH"]):
            if  signals["WST_BUY TOUCH"]:
                msg = f"🔔 WST - {latest['date'].date()}, {symbol}, low {latest['low']}, close {latest['close']}, ST = {latest['Supertrend']}\n"
                #msg = f"🔔 {symbol} (Weekly)\nDate: {latest['date'].date()}\n"
                if signals["WST_BUY TOUCH"]:
                    msg += "✅ BUY WST TOUCH"
                # elif signals["WST_BUY proximity"]:
                #     msg += "⚠️ BUY Proximity to Supertrend"
                # elif signals["WST_SELL rejection"]:
                #     msg += "⚠️ SELL rejection near Supertrend"
                # elif signals["WST_SELL proximity"]:
                #     msg += "⚠️ SELL Proximity to Supertrend"
                send_telegram_alert(msg, bot_token, chat_id)
                alerts.append(symbol)
        else:
            print("Weekly df has insufficient rows for Supertrend calculation")

    print(f"✅ Alerts sent for: {alerts}")
    if alerts:
        send_telegram_alert("*****  WST Completed ********", bot_token, chat_id)
    else :
        send_telegram_alert("*****  No WST signals *********", bot_token, chat_id)
    subprocess.run(["/Users/nags/Data/Alerts/colab/venv/bin/python", "/Users/nags/Data/Alerts/colab/DhDtest.py"])
if __name__ == "__main__":
    main()
