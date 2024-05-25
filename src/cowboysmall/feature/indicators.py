
import ta


INDICATORS = [
    "NSEI_RSI", "DJI_RSI", "NSEI_ROC", "DJI_ROC", "NSEI_AWE", "DJI_AWE", 
    "NSEI_KAM", "DJI_KAM", "NSEI_TSI", "DJI_TSI", "NSEI_VPT", "DJI_VPT", 
    "NSEI_ULC", "DJI_ULC", "NSEI_SMA", "DJI_SMA", "NSEI_EMA", "DJI_EMA"
]


def get_indicators(data):
    data["NSEI_RSI"] = ta.momentum.rsi(data["NSEI_CLOSE"])
    data["DJI_RSI"]  = ta.momentum.rsi(data["DJI_CLOSE"])

    data["NSEI_ROC"] = ta.momentum.roc(data["NSEI_CLOSE"])
    data["DJI_ROC"]  = ta.momentum.roc(data["DJI_CLOSE"])

    data["NSEI_AWE"] = ta.momentum.awesome_oscillator(data["NSEI_HIGH"], data["NSEI_LOW"])
    data["DJI_AWE"]  = ta.momentum.awesome_oscillator(data["DJI_HIGH"], data["NSEI_LOW"])

    data["NSEI_KAM"] = ta.momentum.kama(data["NSEI_CLOSE"])
    data["DJI_KAM"]  = ta.momentum.kama(data["DJI_CLOSE"])

    data["NSEI_TSI"] = ta.momentum.tsi(data["NSEI_CLOSE"])
    data["DJI_TSI"]  = ta.momentum.tsi(data["DJI_CLOSE"])


    data["NSEI_VPT"] = ta.volume.volume_price_trend(data["NSEI_CLOSE"], data["NSEI_VOLUME"])
    data["DJI_VPT"]  = ta.volume.volume_price_trend(data["DJI_CLOSE"], data["DJI_VOLUME"])


    data["NSEI_ULC"] = ta.volatility.ulcer_index(data["NSEI_CLOSE"])
    data["DJI_ULC"]  = ta.volatility.ulcer_index(data["DJI_CLOSE"])


    data["NSEI_SMA"] = ta.trend.sma_indicator(data["NSEI_CLOSE"])
    data["DJI_SMA"]  = ta.trend.sma_indicator(data["DJI_CLOSE"])

    data["NSEI_EMA"] = ta.trend.ema_indicator(data["NSEI_CLOSE"])
    data["DJI_EMA"]  = ta.trend.ema_indicator(data["DJI_CLOSE"])

    return data



# this is no longer used as I am using the ta library for indicator 
# creation - I leave it here for the sake of interest

def calculate_rsi(values, window = 14):
    delta_up = values.diff()
    delta_dn = delta_up.copy()

    delta_up[delta_up < 0] = 0
    delta_dn[delta_dn > 0] = 0

    mean_up = delta_up.rolling(window).mean()
    mean_dn = delta_dn.rolling(window).mean().abs()

    return (mean_up / (mean_up + mean_dn)) * 100
