# market_data/services/indicators.py - À CRÉER

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = calculate_ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calculate_bollinger(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma = calculate_sma(series, period)
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule tous les indicateurs sur un DataFrame OHLCV"""
    result = pd.DataFrame(index=df.index)

    # Moyennes mobiles
    result['sma_20'] = calculate_sma(df['close'], 20)
    result['sma_50'] = calculate_sma(df['close'], 50)
    result['sma_200'] = calculate_sma(df['close'], 200)
    result['ema_12'] = calculate_ema(df['close'], 12)
    result['ema_26'] = calculate_ema(df['close'], 26)

    # RSI
    result['rsi_14'] = calculate_rsi(df['close'], 14)

    # MACD
    macd, signal, hist = calculate_macd(df['close'])
    result['macd'] = macd
    result['macd_signal'] = signal
    result['macd_hist'] = hist

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = calculate_bollinger(df['close'])
    result['bb_upper'] = bb_upper
    result['bb_middle'] = bb_mid
    result['bb_lower'] = bb_lower

    # ATR
    if all(c in df.columns for c in ['high', 'low', 'close']):
        result['atr_14'] = calculate_atr(df['high'], df['low'], df['close'])

    # Volume
    if 'volume' in df.columns:
        result['volume_sma_20'] = calculate_sma(df['volume'], 20)

    return result