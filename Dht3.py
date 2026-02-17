import pandas as pd
import numpy as np
import requests
from datetime import datetime
import json
import subprocess

# This  script uses monthly narrow cpr, checks if daily candle cross quarterly, monthly levels
# ========== CONFIG ==========
CONFIG_FILE = "/users/Nags/Data/Alerts/colab/config.json"
INPUT_FILE = "/users/Nags/Data/Alerts/colab/ohlcv_output_today.xlsx"
FO_FILE = "/users/Nags/Data/Alerts/colab/stockfo.xlsx"
SYMBOL_COL = "symbol"

with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

BOT_TOKEN = config["TELEGRAM_BOT_TOKEN"]
CHAT_ID = config["TELEGRAM_CHAT_ID"]

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    requests.post(url, data=payload)

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

    # Label pin bars
    df1['PinBar'] = None
    df1.loc[bullish, 'PinBar'] = 'Bullish'
    df1.loc[bearish, 'PinBar'] = 'Bearish'

    return df1

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
def check_supertrend_signals(row, level_name='Supertrend', proximity_pct=25):
    level = row.get(level_name)
    atr = row.get('ATR')
    
    if pd.isna(level) or pd.isna(atr): return {"MST_BUY Touch": False, "MST_BUY bounce": False, "MST_BUY proximity": False}
    #print(f"  Level {level} ATR {atr}")
    proximity = atr * proximity_pct / 100
    open_, high, low, close = row['open'], row['high'], row['low'], row['close']
    body = abs(close - open_)
    lower_wick = min(open_, close) - low
    upper_wick = high - max(open_, close)
    range1 = abs(high - low)

    bproximity = low > level and low < (level + proximity)
    sproximity = high < level and high > (level - proximity)
    btouch = low <= level <= high and close > level
    stouch = high >= level >= low and close < level

    bbounce = (bproximity or btouch) and (close > level) and (
        (lower_wick > 2 * body and upper_wick < body and body / range1 < 0.5) or
        ((body > 0.5 * range1) or (high - level) > 0.5 * range1)
    )

    srejection = (sproximity or stouch) and (close < level) and (
        (upper_wick > 2 * body and lower_wick < body and body / range1 < 0.5) or
        ((body > 0.5 * range1) or (level - low) > 0.5 * range1)
    )
   
    return {
        "MST_BUY Touch" : btouch or bbounce or bproximity,

        # "MST_BUY bounce": bbounce,
        # "MST_BUY proximity": bproximity,
         "MST_SELL Touch": srejection or sproximity or stouch,  # optional
        # "MST_SELL proximity": sproximity
    }

# ========== CPR CALCULATION ==========
def calculate_cpr_levels(df):
    df = df.copy()
    df["bc"] = (df["high"] + df["low"]) / 2
    df["pivot"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tc"] = 2 * df["pivot"] - df["bc"]
    df["tc"], df["bc"] = df[["tc", "bc"]].max(axis=1), df[["tc", "bc"]].min(axis=1)

    df["r1"] = 2 * df["pivot"] - df["low"]
    df["s1"] = 2 * df["pivot"] - df["high"]
    df["r2"] = df["pivot"] + (df["r1"] - df["s1"])
    df["s2"] = df["pivot"] - (df["r1"] - df["s1"])

    return df
#============ 200 EMA calculation on daily ============
def calculate_ema(series, period=200):
    return series.ewm(span=period, adjust=False).mean()

# ========== DAILY CANDLE SIGNAL LOGIC ==========
def check_cpr_signals(row, level_name, proximity_pct=0.1):
    level = row[level_name]
    if pd.isna(level) or level == 1: return {}

    # Current Daily OHLC
    o, h, l, c = row['open'], row['high'], row['low'], row['close']
    

    # Your Specific Touch Logic
    # btouch: low is below or equal to level, but close is above, and low MUST be strictly less than level
    btouch = l <= level <= h  and c > level
    stouch = h >= level >= l  and c < level

    # Optional Bounce/Rejection logic (commented as per your request but available)
    # bbounce = (btouch) and (c > level) and ((lower_wick > 2 * body and upper_wick < body and body / range1 < 0.5))
    
    signals = {}
    if btouch: signals[f"{level_name}_BUY touch"] = True
    if stouch: signals[f"{level_name}_SELL touch"] = True
    
    return signals

def calculate_vwap(df):
    """
    Calculates VWAP for a given OHLCV DataFrame.
    
    Parameters:
    df : pd.DataFrame
        Must contain columns: ['high','low','close','volume']
        This can be resampled daily, weekly, monthly, etc.
    
    Returns:
    float
        VWAP for the provided interval.
    """
    if df.empty or 'volume' not in df.columns:
        return None
    
    # Typical Price
    df['Typical_Price'] = (df['high'] + df['low'] + df['close']) / 3
    df['TPV'] = df['Typical_Price'] * df['volume']

    # VWAP
    # total_volume = df['volume']
    # if total_volume == 0:
    #     return None
    vwap = df['TPV'] / df['volume']
    return vwap

def calculate_macd(df, l) :
    fast=12
    slow=l
    signal=9

    # Ensure 'close' column exists
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")

    # EMAs
    df['EMA_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['close'].ewm(span=slow, adjust=False).mean()

    # MACD Line
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']

    # Signal Line
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()

    # Histogram
    df['Histogram'] = df['MACD'] - df['Signal']
    
    return df

def detect_macd_crosses(df):
    df['prev_MACD'] = df['MACD'].shift(1)
    df['prev_Signal'] = df['Signal'].shift(1)

    df['Bullish_Cross'] = (df['prev_MACD'] < df['prev_Signal']) & (df['MACD'] > df['Signal'])
    df['Bearish_Cross'] = (df['prev_MACD'] > df['prev_Signal']) & (df['MACD'] < df['Signal'])

    return df

def evaluate_trade_setup(row, m_st, w_st, h1_macd_cross=None, h1_st_buy=None):
    """
    Evaluates the Buy/Short/No-Trade decision tree based on MTF data.
    """
    # Trend Alignment Data
    m_trend = m_st.iloc[-1]['Trend'] == 'Uptrend'
    w_trend = w_st.iloc[-1]['Trend'] == 'Uptrend'
    price = row['close']
    ema_200 = row['ema_200']
    
    # Structure Data (CPR)
    pivot = row['m_pivot']
    tc = row['m_tc']
    bc = row['m_bc']
    
    # Value Data (VWAP)
    vwap = row.get('VWAP', 0)

    # --- NO-TRADE CONDITIONS (Check first) ---
    if m_trend != w_trend:
        return "NO-TRADE (Trend Mismatch)"
    if abs(price - vwap) < (price * 0.001): # Arbitrary small buffer
        return "NO-TRADE (At VWAP - No Edge)"
    if bc <= price <= tc and abs(price - vwap) < (price * 0.002):
        return "NO-TRADE (Chopzone: Inside CPR + VWAP)"

    # --- BUY SETUP (LONG) ---
    if m_trend and w_trend and price > ema_200:
        # Value Check: Discounted fills
        if price <= vwap:
            # Structure Check
            if price >= bc: # Above or Inside CPR
                # Trigger Check
                if h1_macd_cross == 'Bullish' or h1_st_buy or price > vwap:
                    return "BUY SETUP (LONG)"

    # --- SHORT SETUP (SELL) ---
    if not m_trend and not w_trend and price < ema_200:
        # Value Check: Premium supply
        if price >= vwap:
            # Structure Check
            if price <= tc: # Below or Inside CPR
                # Trigger Check
                if h1_macd_cross == 'Bearish' or not h1_st_buy or price < vwap:
                    return "SHORT SETUP (SELL)"

    return "NO-TRADE (Criteria Not Met)"

# ========== MAIN ==========
def main():
    # --- 1. Initialize Collections ---
    mst_alerts = []
    wst_alerts = []
    dst_alerts = []
    mpin_alerts = []
    wpin_alerts = []
    mvolume_alerts = []
    wvolume_alerts = []
    mcpr_alerts = []
    wcpr_alerts = []
    dcpr_alerts = []
    macdalerts = []
    mst_alerts1 = []
    wst_alerts1 = []
    dst_alerts1 = []

    # mst_switch = 0
    # wst_switch = 0
    # dst_switch = 0
    # mpin_switch = 0
    # wpin_switch = 0
    # mvolume_switch = 0
    # wvolume_switch = 0
    # mcpr_switch = 0
    # wcpr_switch = 0
    # dcpr_switch = 0


    mst_switch = 1
    wst_switch = 1
    dst_switch = 1
    mpin_switch = 1
    wpin_switch = 1
    mvolume_switch = 1
    wvolume_switch = 1
    mcpr_switch = 1
    wcpr_switch = 1
    dcpr_switch = 1


    fo_df = pd.read_excel(FO_FILE)
    fo_list = set(fo_df[SYMBOL_COL].str.upper().str.strip())

    df = pd.read_excel(INPUT_FILE)
    df[SYMBOL_COL] = df[SYMBOL_COL].str.upper().str.strip()
    #df = df[df[SYMBOL_COL].isin(fo_list)]
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values([SYMBOL_COL, 'date'])

    #send_telegram_alert("🔍 *MTF Scan: Monthly & Quarterly Levels + Daily Candle Signal*")
    # --- NEW: DAILY 200 EMA ---
        # Calculate this on the raw daily data before resampling
    for symbol, sdf in df.groupby(SYMBOL_COL):
        label = ""
        if symbol in fo_list: label = "FUT" 
        sdf_idx = sdf.set_index('date')
        if len(sdf_idx) >= 200:
            sdf_idx['ema_200'] = calculate_ema(sdf_idx['close'], 200)
        else:
            sdf_idx['ema_200'] = np.nan

        # ---  Check for SUPERTREND LEVELS ---
        # y_df = sdf_idx.resample('YE').agg({'open' : 'first', 'high': 'max', 'low': 'min', 'close': 'last' , 'volume': 'sum'}).dropna().reset_index()
        # hf_df = sdf_idx.resample('2QE-JUN').agg({'open' : 'first', 'high': 'max', 'low': 'min', 'close': 'last' , 'volume': 'sum'}).dropna().reset_index()
        m_df = sdf_idx.resample('ME').agg({'open' : 'first', 'high': 'max', 'low': 'min', 'close': 'last' , 'volume': 'sum'}).dropna().reset_index()
        q_df = sdf_idx.resample('QS').agg({'open' : 'first', 'high': 'max', 'low': 'min', 'close': 'last' , 'volume': 'sum'}).dropna().reset_index()
        w_df = sdf_idx.resample('W').agg({ 'open' : 'first', 'high': 'max', 'low': 'min', 'close': 'last' , 'volume': 'sum'}).dropna().reset_index() 
        d_df = sdf_idx.resample('D').agg({ 'open' : 'first', 'high': 'max', 'low': 'min', 'close': 'last' , 'volume': 'sum'}).dropna().reset_index()
# --- MACD Calculation with Safety Check ---
        if len(m_df) >= 21:
            l = len(m_df)
            m_df = calculate_macd(m_df, l)
            m_df = detect_macd_crosses(m_df)
        else:
            m_df['MACD'] = m_df['Signal'] = m_df['Bullish_Cross'] = m_df['Bearish_Cross'] = False

        if len(w_df) >= 26:
            w_df = calculate_macd(w_df, l=26)
            w_df = detect_macd_crosses(w_df)
        else:
            w_df['MACD'] = w_df['Signal'] = w_df['Bearish_Cross'] = w_df['Bullish_Cross'] = False

        d_df = calculate_macd(d_df, l=26)
        d_df = detect_macd_crosses(d_df)
        #print(f"{symbol} Daily MACD = {m_df.iloc[-1]['MACD']} Signal = {m_df.iloc[-1]['Signal']}")
        # if m_df['Bullish_Cross'].iloc[-1] and w_df['Bullish_Cross'].iloc[-1] :
        #     print(f"{symbol} Monthly MACD = {m_df.iloc[-1]['MACD']} Signal = {m_df.iloc[-1]['Signal']}")
        #     print(f"Weekly MACD = {w_df.iloc[-1]['MACD']} Signal = {w_df.iloc[-1]['Signal']}")
        #     macdalerts.append(f" {symbol} {label} BULLISH cross (close: {m_df.iloc[-1]['close']:.2f})")

        # if m_df['Bearish_Cross'].iloc[-1] & w_df['Bearish_Cross'].iloc[-1] :

        #     print(f"{symbol} Monthly MACD = {m_df.iloc[-1]['MACD']} Signal = {m_df.iloc[-1]['Signal']}")
        #     print(f"Weekly MACD = {w_df.iloc[-1]['MACD']} Signal = {w_df.iloc[-1]['Signal']}")
        #     macdalerts.append(f" {symbol} {label} BEARISH cross (close: {m_df.iloc[-1]['close']:.2f})")

        if len(m_df) < 2: continue
        m_st = compute_supertrend(m_df)
        mlatest = d_df.iloc[-1].copy()
        mlatest['Supertrend'] = m_st.iloc[-1]['Supertrend']
        mlatest['ATR'] = m_st.iloc[-1]['ATR']
        msignals = check_supertrend_signals(mlatest)
        if mst_switch == 1 and msignals["MST_BUY Touch"]:
            print(f"MST BUY PRESENT")
            mst_alerts.append(f"🔔 {symbol}  mst {label} (Close: {mlatest['close']:.2f}, ST: {mlatest['Supertrend']:.2f})")
        if mst_switch == 1 and msignals["MST_SELL Touch"]:
            print(f"MST SELL PRESENT")
            mst_alerts1.append(f"🔔 {symbol}  mst {label} (Close: {mlatest['close']:.2f}, ST: {mlatest['Supertrend']:.2f})")
          

        w_st = compute_supertrend(w_df)
        wlatest = d_df.iloc[-1].copy()
        wlatest['Supertrend'] = w_st.iloc[-1]['Supertrend']
        wlatest['ATR'] = w_st.iloc[-1]['ATR']
        # wlatest = d_df.iloc[-1]
        # print(f"{symbol} 1")
        # print(w_st.iloc[-1].to_string())
        # print(f"WLATEST ")
        # print(wlatest.to_string())
        wsignals = check_supertrend_signals(wlatest)
        if label =="FUT" and wst_switch == 1 and wsignals["MST_BUY Touch"]:
            print(f"WST BUY PRESENT")
    
            wst_alerts.append(f"🔔{symbol} wst {label} (Close: {wlatest['close']:.2f}, ST: {wlatest['Supertrend']:.2f})")
     
        if label =="FUT" and wst_switch == 1 and wsignals["MST_SELL Touch"]:
            print(f"MST SELL PRESENT")
            wst_alerts1.append(f"🔔 {symbol}  mst {label} (Close: {mlatest['close']:.2f}, ST: {mlatest['Supertrend']:.2f})")

       

        #-------------- Volume SPike ------------------- 
        if not m_df.empty and len(m_df) > 13:
            mrecent = m_df.tail(13)
            today_volume = round(mrecent.iloc[-1]["volume"], 2)
            avg_past_12 = mrecent["volume"].tail(13).mean()
            vol_spike = today_volume > avg_past_12  * 5  # You can apply a multiplier like avg_past_12 * 2

            if mvolume_switch == 1 and vol_spike:
                print(f"Monthly Volume found")
                mvolume_alerts.append(f"📊 {symbol}, {label} Close: {mrecent.iloc[-1]['close']:.2f}, 🔊 Volume: {today_volume}, 📊 12M Avg Volume: {round(avg_past_12),2}")

                #     msg = (
                #     f"📌 Monthly Volume spike detected:\n"
                #     f"📅 {mrecent.iloc[-1]['date'].date()}, "
                #     f"💹 {symbol}, "
                #     f"📈 Close: {mrecent.iloc[-1]['close']}, "
                #     f"🔊 Volume: {today_volume}, "
                #     f"📊 12M Avg Volume: {round(avg_past_12)}"
                # )
                # send_telegram_alert(msg)

        if not w_df.empty and len(w_df) > 13:
            wrecent = w_df.tail(13)
            today_volume = round(wrecent.iloc[-1]["volume"])
            avg_past_12 = wrecent["volume"].tail(13).mean()
            vol_spike = today_volume > avg_past_12  * 8  # You can apply a multiplier like avg_past_12 * 2

            if label =="FUT" and wvolume_switch == 1 and vol_spike:
                print(f"Weekly Volume found")
                wvolume_alerts.append(f"📊 {symbol}, {label} Close: {mrecent.iloc[-1]['close']:.2f}, 🔊 Volume: {today_volume}, 📊 12M Avg Volume: {round(avg_past_12)}")

                        # msg = (
                        #                     f"📌 Monthly Volume spike detected:\n"
                        #                     f"📅 {wrecent.iloc[-1]['date'].date()}, "
                        #                     f"💹 {symbol}, "
                        #                     f"📈 Close: {wrecent.iloc[-1]['close']}, "
                        #                     f"🔊 Volume: {today_volume}, "
                        #                     f"📊 12M Avg Volume: {round(avg_past_12)}"
                        #                 )
                #send_telegram_alert(msg)



        #---------- 1. MONTHLY LEVELS -----------------
    # if symbol in fo_list:
         #------------------------PINBAR LOGIC -------
      
        today = datetime.today().day
        # --- WEEKLY PIN BAR (ONLY ON FRIDAY) ---
        
        
        d_st = compute_supertrend(d_df)
        dlatest = d_df.iloc[-1]
        dsignals = check_supertrend_signals(dlatest)
        if label =="FUT" and dst_switch == 1 and dsignals["MST_BUY Touch"]:
            print(f"DST BUY PRESENT")
            dst_alerts.append(f"🔔 {symbol} {label} dst  (Close: {dlatest['close']:.2f}, ST: {dlatest['Supertrend']:.2f})")
        if label =="FUT" and dst_switch == 1 and dsignals["MST_SELL Touch"]:
            print(f"DST SELL PRESENT")
            dst_alerts1.append(f"🔔 {symbol} {label} dst  (Close: {dlatest['close']:.2f}, ST: {dlatest['Supertrend']:.2f})")
        
        m_df['MVWAP'] = calculate_vwap(m_df)
        w_df['WVWAP'] = calculate_vwap(w_df)
        d_df['DVWAP'] = calculate_vwap(d_df)



        #print({symbol}, m_df[['date','high', 'low', 'close' , 'volume', 'Typical_Price', 'TPV', 'MVWAP']])

        if today > 25:
            mplatest = detect_pin_bars(m_df)
            mpinbar_type = mplatest.iloc[-1]['PinBar']
            if mpin_switch == 1 and mpinbar_type in ['Bullish','BEARISH']:
                mpin_alerts.append(f"📌 {symbol}, {label} mpin  📈 Close: {mplatest.iloc[-1]['close']:.2f}")
          

        # Check if today is Friday (4)
        current_date = datetime.now()
        if current_date.weekday() == 4:
            # Execute only from 10th day of the month
            wplatest = detect_pin_bars(w_df)
            wpinbar_type = wplatest.iloc[-1]['PinBar']
            if wpin_switch == 1 and wpinbar_type in ['Bullish','BEARISH']:
                wpin_alerts.append(f"📌 {symbol}, {label} wpin  📈 Close: {wplatest.iloc[-1]['close']:.2f}")
                    #                     msg = f"📌 WEEKLY Pin Bar Detected: {wpinbar_type}\n📅 {wplatest.iloc[0]['date'].date()}, 💹 {symbol}, 📈 Close: {wplatest.iloc[0]['close']}"
                    # send_telegram_alert(msg)
                    # print(f"WPin alert sent for {symbol}")

        m_cpr = calculate_cpr_levels(m_df)
        # Calculate width for all months: (TC - BC)
        # m_cpr['width_val'] = (m_cpr['tc'] - m_cpr['bc']).abs()
        
        # # Calculate 5-month average width (excluding the very latest/current month)
        # # We look at indices -6 to -2 (the 5 months prior to the signal month)
        # m_avg_width_5 = m_cpr['width_val'].iloc[-6:-1].mean()



        m_width = (abs(m_cpr.iloc[-2]['tc'] - m_cpr.iloc[-2]['bc']) / m_cpr.iloc[-2]['pivot']) * 100
        # ismonthlynarrow = m_width < (m_avg_width_5 * 0.25)



        m_cpr_shifted = m_cpr.shift(1).rename(columns={c: f"m_{c}" for c in m_cpr.columns if c != 'date'})
        # Create join key: Add 1 month to the original date
        m_cpr_shifted['m_join_key'] = (m_cpr_shifted['date'] + pd.DateOffset(months=1)).dt.to_period('M')

        # --- 2. QUARTERLY LEVELS ---
       
        if len(q_df) < 2: continue
        q_cpr = calculate_cpr_levels(q_df)

        
        q_cpr_shifted = q_cpr.shift(1).rename(columns={c: f"q_{c}" for c in q_cpr.columns if c != 'date'})
        # Create join key: Add 3 months to the original date
        q_cpr_shifted['q_join_key'] = (q_cpr_shifted['date'] + pd.DateOffset(months=3)).dt.to_period('Q')

        # --- 3. WEEKLY LEVELS ---
        
        if len(w_df) < 2: continue
        w_cpr = calculate_cpr_levels(w_df)
        

        # Decide width based on Friday or Mid-Week
        latest_date = sdf_idx.index[-1]
        day_of_week = latest_date.weekday()
        row_for_width = w_cpr.iloc[-1] if day_of_week >= 4 else w_cpr.iloc[-2]
        w_width = (abs(row_for_width['tc'] - row_for_width['bc']) / row_for_width['pivot']) * 100


        # --- 4. DAILY LEVELS ---
        
        if len(d_df) < 2: continue
        d_cpr = calculate_cpr_levels(d_df)
        
        d_width = (abs(d_cpr.iloc[-1]['tc'] - d_cpr.iloc[-1]['bc']) / d_cpr.iloc[-1]['pivot']) * 100


        # # Calculate width for all days: (TC - BC)
        # d_cpr['width_val'] = (d_cpr['tc'] - d_cpr['bc']).abs()
        
        # # Calculate 5-month average width (excluding the very latest/current month)
        # # We look at indices -6 to -2 (the 5 months prior to the signal month)
        # d_avg_width_5 = d_cpr['width_val'].iloc[-6:-1].mean()



        # d_width = (abs(d_cpr.iloc[-2]['tc'] - d_cpr.iloc[-2]['bc']) / d_cpr.iloc[-2]['pivot']) * 100
        # isdailynarrow = d_width < (d_avg_width_5 * 0.25)  # 45% narrow than average





        # --- 5. MERGE ALL INTO DAILY DATAFRAME ---
        sdf_with_ema = sdf_idx.reset_index()
        sdf_with_ema['m_key'] = sdf_with_ema['date'].dt.to_period('M')
        sdf_with_ema['q_key'] = sdf_with_ema['date'].dt.to_period('Q')
        
        # sdf['m_key'] = sdf['date'].dt.to_period('M')
        # sdf['q_key'] = sdf['date'].dt.to_period('Q')
        
        # Drop 'date' from CPR frames before merging to avoid 'date_x/y' errors
        m_merge_ready = m_cpr_shifted.drop(columns=['date'], errors='ignore')
        q_merge_ready = q_cpr_shifted.drop(columns=['date'], errors='ignore')
        
        # Now merge using the dataframe that has the EMA column
        
        merged = sdf_with_ema.merge(m_merge_ready, left_on='m_key', right_on='m_join_key', how='left')
        merged = merged.merge(q_merge_ready, left_on='q_key', right_on='q_join_key', how='left')

        
        # merged = sdf.merge(m_merge_ready, left_on='m_key', right_on='m_join_key', how='left')
        # merged = merged.merge(q_merge_ready, left_on='q_key', right_on='q_join_key', how='left')

        # Last day candle
        last_day = merged.iloc[-1]
        trade_decision = evaluate_trade_setup(last_day, m_st, w_st)

        if "BUY" in trade_decision:
            buy_msg = f"🚀 <b>BUY ALERT: {symbol}</b>\nTrend: Bullish Alignment\nTarget: Weekly CPR/Quarterly VWAP"
            # send_telegram_alert(buy_msg)
        elif "SHORT" in trade_decision:
            short_msg = f"🔻 <b>SHORT ALERT: {symbol}</b>\nTrend: Bearish Alignment\nTarget: Prev Day VWAP"
            # send_telegram_alert(short_msg)




        m_levels = ['m_pivot', 'm_tc', 'm_bc', 'm_r1', 'm_s1', 'm_r2', 'm_s2']
        q_levels = ['q_pivot', 'q_tc', 'q_bc', 'q_r1', 'q_s1', 'q_r2', 'q_s2']
        
        # Debug Print

        # print(f"\nSTOCK: {symbol} | CLOSE: {last_day['close']:.2f} | LOW: {last_day['low']:.2f} | HIGH: {last_day['high']:.2f}")
        # print(f"  {m_width:.2f}%, Monthly Pivot: {last_day['m_pivot']:.2f} (TC: {last_day['m_tc']:.2f}, BC: {last_day['m_bc']:.2f})")
        # print(f"  Quarterly Pivot: {last_day['q_pivot']:.2f} (TC: {last_day['q_tc']:.2f}, BC: {last_day['q_bc']:.2f})")
        # print(f"  {w_width:.2f}%, Weekly -- (TC: {row_for_width['tc']:.2f}, Pivot: {row_for_width['pivot']:.2f} ,BC: {row_for_width['bc']:.2f})")
        # print(f"  {d_width:.2f}%, Daily -- (TC: {d_cpr.iloc[-1]['tc']:.2f}, Pivot: {d_cpr.iloc[-1]['pivot']:.2f} ,BC: {d_cpr.iloc[-1]['bc']:.2f})")
        
        all_signals = []
        
        # Check Monthly levels using Daily Candle
        for lvl in m_levels:
            sig = check_cpr_signals(last_day, lvl)
            if sig: all_signals.extend(sig.keys())
            
        # Check Quarterly levels using Daily Candle
        for lvl in q_levels:
            sig = check_cpr_signals(last_day, lvl)
            if sig: all_signals.extend(sig.keys())

        trend = "BULLISH  " if last_day['ema_200'] < last_day ['close'] else "BEARISH"
        
        #fo_tag = " <b>(Futures)</b>" if symbol in fo_list else ""

        # ... (rest of the logic for merging and signal checking) ...

        #Logic: If Monthly Width is Narrow AND any signal is triggered
        if   (label == "FUT" and mcpr_switch == 1 and m_width < 0.2) or (label == "FUT" and mcpr_switch == 1 and m_width < 0.2 and all_signals) :  # and w_width <= 0.45 and d_width <= 0.40 and all_signals:
            msg = (f"✅{symbol} {label} Monthly nCPR-{m_width:.2f}%, TC: {last_day['m_tc']:.2f}, Pivot: {last_day['m_pivot']:.2f}, BC: {last_day['m_bc']:.2f}\n"
                   f"trend is {trend}---Close = {last_day['close']:.2f}\n"
                   f" *Signals:* {', '.join(all_signals)}")
            mcpr_alerts.append(msg)
            if d_width < 0.1 and all_signals:
                msg = (f" ✅{symbol} {label} Daily nCPR- {d_width:.2f}%TC: {d_cpr.iloc[-1]['tc']:.2f}, Pivot: {d_cpr.iloc[-1]['pivot']:.2f} ,BC: {d_cpr.iloc[-1]['bc']:.2f}\n" 
                f" *Signals:* {', '.join(all_signals)}")
                dcpr_alerts.append(msg)

            #send_telegram_alert(msg)
            print(f"✅ Monthly CPR Alert sent for {symbol}")
        if (label =="FUT" and wcpr_switch == 1 and w_width < 0.1 and all_signals) or (label =="FUT" and wcpr_switch == 1 and w_width < 0.15):
            msg = (
               #f"📏 *M Width:* {m_width:.2f}%----*W Width {w_width:.2f}%\n"
               f"✅{symbol} {label} Weekly nCPR- {w_width:.2f}%, TC: {row_for_width['tc']:.2f}, Pivot: {row_for_width['pivot']:.2f} ,BC: {row_for_width['bc']:.2f}\n"
               f" *Signals:* {', '.join(all_signals)}")
            wcpr_alerts.append(msg)
            # send_telegram_alert(msg)
            print(f"✅ Weekly CPR Alert sent for {symbol}")

        # if label =="FUT" and dcpr_switch == 1 and (m_width < 0.3 or w_width < 0.15 ) and d_width < 0.1 and all_signals:
        #     msg = (f" ✅{symbol} {label} Daily nCPR- {d_width:.2f}%TC: {d_cpr.iloc[-1]['tc']:.2f}, Pivot: {d_cpr.iloc[-1]['pivot']:.2f} ,BC: {d_cpr.iloc[-1]['bc']:.2f}\n" 
        #         f" *Signals:* {', '.join(all_signals)}")
        #     dcpr_alerts.append(msg)
        #     # send_telegram_alert(msg)
        #     print(f"✅ Daily CPR Alert sent for {symbol}")


    # ---Send Consolidated Alerts at the End ---
    tod = datetime.today().day
    cdate = datetime.now()

    if macdalerts:
        print(f" MACD cross alerts")
        send_telegram_alert("MACD Alerts: \n"+"\n".join(macdalerts))

    # if dcpr_alerts:
        print(f"Daily FUT narrow CPR ALERTS are {dcpr_alerts}")
        send_telegram_alert(("🚀 <b> Daily nCPR Alerts:</b>\n" + "\n".join(dcpr_alerts)))
    else: 
        send_telegram_alert("NO Daily nCPR alerts")

    if dst_alerts:
        print (f"DST BUY FuT Alerts are {dst_alerts}")
        send_telegram_alert(("🚀 <b> DAILY BUY SUPERTREND (FUT) ALERTS:</b>\n" + "\n".join(dst_alerts)))
    else: 
        send_telegram_alert("NO DST BUY alerts")
    if dst_alerts1:
        print (f"DST SELL FuT Alerts are {dst_alerts}")
        send_telegram_alert(("🚀 <b> DAILY SELL SUPERTREND (FUT) ALERTS:</b>\n" + "\n".join(dst_alerts)))
    else: 
        send_telegram_alert("NO DST SELL alerts")


    if wst_alerts:
        print (f"WST BUY Alerts are {wst_alerts}")
        send_telegram_alert(("🚀 <b> WEEKLY BUY SUPERTREND ALERTS:</b>\n" + "\n".join(wst_alerts)))
    else: 
        send_telegram_alert("NO WST BUY alerts")
    if wst_alerts1:
        print (f"WST SELL Alerts are {wst_alerts}")
        send_telegram_alert(("🚀 <b> WEEKLY SELL SUPERTREND ALERTS:</b>\n" + "\n".join(wst_alerts)))
    else: 
        send_telegram_alert("NO WST SELL alerts")

    if mst_alerts:
        print (f"MST BUY Alerts are {mst_alerts}")
        send_telegram_alert(("🚀 <b> MONTHLY BUY SUPERTREND ALERTS:</b>\n" + "\n".join(mst_alerts)))
    else: 
        send_telegram_alert("No MST BUY alerts")

    if mst_alerts1:
        print (f"MST SELL Alerts are {mst_alerts}")
        send_telegram_alert(("🚀 <b> MONTHLY SELL SUPERTREND ALERTS:</b>\n" + "\n".join(mst_alerts)))
    else: 
        send_telegram_alert("No MST SELL alerts")


    if dcpr_alerts:
        print(f"Daily FUT narrow CPR ALERTS are {wcpr_alerts}")
        send_telegram_alert(("🚀 <b> Daily nCPR Alerts:</b>\n" + "\n".join(dcpr_alerts)))
    else:
        send_telegram_alert("NO daily nCPR alerts")
    if cdate.weekday() in (0, 1, 4):
        if wcpr_alerts:
            print(f"Weekly FUT narrow CPR ALERTS are {wcpr_alerts}")
            send_telegram_alert(("🚀 <b> Weekly nCPR Alerts:</b>\n" + "\n".join(wcpr_alerts)))
        else: 
            send_telegram_alert("NO Weekly nCPR alerts")
    else:
        send_telegram_alert("Weekly nCPR sent already")


    if cdate.weekday() in (0, 1, 4):
        if mcpr_alerts:
            print(f"Monthly FUT narrow CPR ALERTS are {mcpr_alerts}")
            send_telegram_alert(("🚀 <b> MONTHLY nCPR Alerts:</b>\n" + "\n".join(mcpr_alerts)))
        else: 
            send_telegram_alert("NO Monthly nCPR alerts")
    else:
        send_telegram_alert("MONTHLY nCPR sent already")

    if mvolume_alerts:
        print (f"Monthly Volume SPike Alerts are {mvolume_alerts}")
        send_telegram_alert(("🔊  <b> Monthly Volume Spike ALERTS:</b>\n" + "\n".join(mvolume_alerts)))
    else: 
        send_telegram_alert("NO Monthly Volume alerts")

    if wvolume_alerts:
        print (f"Weekly Volume Spike Alerts are {wvolume_alerts}")
        send_telegram_alert(("🔊  <b> Weekly Volume Spike ALERTS:</b>\n" + "\n".join(wvolume_alerts)))
    else: 
        send_telegram_alert("NO Weekly Volume alerts")
    
    if tod > 25:
        if mpin_alerts:
            print (f"Monthly PIN Alerts are {mpin_alerts}")
            send_telegram_alert(("📌 <b>Monthly PIN Fut ALERTS:</b>\n" + "\n".join(mpin_alerts)))
        else: 
            send_telegram_alert("NO Monthly PIN alerts")
    else:
        send_telegram_alert("MPIN will be checked in last week of the month only")



    if cdate.weekday() == 4:
        if wpin_alerts:
            print (f"Weekly PIN Alerts are {wpin_alerts}")
            send_telegram_alert(("📌  <b> WEEKLY PIN Fut ALERTS:</b>\n" + "\n".join(wpin_alerts)))
        else: 
            send_telegram_alert("NO Weekly PIN alerts")

    else:
        send_telegram_alert("Weekly Pin only on Friday")


    #send_telegram_alert("✅ *Daily MTF Signal Scan Completed.*")
    #subprocess.run(["/Users/nags/Data/Alerts/colab/venv/bin/python", "/Users/nags/Data/Alerts/colab/DhYCPR2.py"])

if __name__ == "__main__":
    main()