
def calculate_rsi(values, window = 14):
    delta_up = values.diff()
    delta_dn = delta_up.copy()

    delta_up[delta_up < 0] = 0
    delta_dn[delta_dn > 0] = 0

    mean_up = delta_up.rolling(window).mean()
    mean_dn = delta_dn.rolling(window).mean().abs()

    return (mean_up / (mean_up + mean_dn)) * 100
