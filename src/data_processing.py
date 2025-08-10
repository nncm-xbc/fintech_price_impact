import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import jax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

trades_df = pd.concat([
    pd.read_csv(file, header=None, names=[
        'trade_id', 'price', 'volume', 'quote_volume', 
        'timestamp', 'is_buyer_maker', 'is_best_match'
    ]) for file in glob.glob(os.path.join("data/binance_raw", "BTCUSDT-trades*.csv"))
], ignore_index=True)

trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], unit='ns')
trades_df['price'] = trades_df['price'].astype(float)
trades_df['volume'] = trades_df['volume'].astype(float)
trades_df['trade_sign'] = trades_df['is_buyer_maker'].apply(lambda x: -1 if x else 1)
trades_df = trades_df.drop(columns=['is_buyer_maker', 'is_best_match'], axis=1)

volumes = trades_df['volume']
print(f"     Median: {np.median(volumes):.6f} BTC")
print(f"     Zero volumes: {np.sum(volumes == 0)} ({100*np.mean(volumes == 0):.2f}%)")
median_volume = np.median(volumes[volumes > 0])
scale_factor = 1.0 / median_volume
trades_clean = trades_df.copy()
trades_clean['volume_normalized'] = trades_clean['volume'] * scale_factor
print("   Scale factor:", scale_factor)
trades_clean['log_volume'] = np.log(trades_clean['volume_normalized'])
trades_clean['sqrt_volume'] = np.sqrt(trades_clean['volume_normalized'])

print(trades_clean.head())
print(trades_clean.shape)
"""
ohlvc_df = pd.concat([
    pd.read_csv(file, header=None, names=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
    ]) for file in glob.glob(os.path.join("data/binance_raw", "BTCUSDT-1s*.csv"))
], ignore_index=True)

ohlvc_df['open_time'] = pd.to_datetime(ohlvc_df['open_time'], unit='ns')
ohlvc_df['close_time'] = pd.to_datetime(ohlvc_df['close_time'], unit='ns')
ohlvc_df['midpoint'] = (ohlvc_df['high'] + ohlvc_df['low'])/2
ohlvc_df['spread'] = 2 * (ohlvc_df['high'] - ohlvc_df['low'])
"""
prices = trades_clean['price'].values
signs = trades_clean['trade_sign'].values
volumes = trades_clean['volume_normalized'].values
log_volumes = trades_clean['log_volume'].values
