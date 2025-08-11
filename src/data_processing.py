import os
import glob
import argparse
import pandas as pd
import numpy as np

#==================================================================================
# Data collection and processing
#==================================================================================

def collect_trades():

    print("Collecting data from raw files:")
    for file in glob.glob(os.path.join("../data/binance_raw", "BTCUSDT-trades*.csv")):
        print(file)

    # concatenate all raw csv files into one dataframe
    trades = pd.concat([
        pd.read_csv(file, header=None, names=[
            'trade_id', 'price', 'volume', 'quote_volume', 
            'timestamp', 'is_buyer_maker', 'is_best_match'
        ]) for file in glob.glob(os.path.join("../data/binance_raw", "BTCUSDT-trades*.csv"))
    ], ignore_index=True)

    # Convert raw features into usable datatypes 
    trades['timestamp'] = pd.to_datetime(trades['timestamp'], unit='ns')
    trades['price'] = trades['price'].astype(float)
    trades['volume'] = trades['volume'].astype(float)
    trades['trade_sign'] = trades['is_buyer_maker'].apply(lambda x: -1 if x else 1)
    trades = trades.drop(columns=['is_buyer_maker', 'is_best_match'], axis=1)

    # Log-normalize the volume to avoid rounding to zero issues
    # since we have originally valid volumes that are very close to 0 (1e-5)
    volumes = trades['volume']
    median_volume = np.median(volumes[volumes > 0])
    scale_factor = 1.0 / median_volume
    trades['volume_normalized'] = trades['volume'] * scale_factor
    trades['log_volume'] = np.log(trades['volume_normalized'])
    trades['sqrt_volume'] = np.sqrt(trades['volume_normalized'])

    trades.to_csv('../data/trades_dense.csv', index=False)

    print(trades.head())
    print(trades.shape)

    return trades

def collect_quotes():
    print("Collecting quotes from raw files:")
    for file in glob.glob(os.path.join("data/binance_raw", "BTCUSDT-1s*.csv")):
        print(file)
    
    ohlvc = pd.concat([
        pd.read_csv(file, header=None, names=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                    'taker_buy_quote_volume', 'ignore'
        ]) for file in glob.glob(os.path.join("../data/binance_raw", "BTCUSDT-1s*.csv"))
    ], ignore_index=True)

    ohlvc['open_time'] = pd.to_datetime(ohlvc['open_time'], unit='ns')
    ohlvc['close_time'] = pd.to_datetime(ohlvc['close_time'], unit='ns')
    ohlvc['midpoint'] = (ohlvc['high'] + ohlvc['low'])/2
    ohlvc['spread'] = 2 * (ohlvc['high'] - ohlvc['low'])
    ohlvc['quote_volume'] = ohlvc['quote_volume'].astype(float)
    ohlvc['open'] = ohlvc['open'].astype(float)
    ohlvc['high'] = ohlvc['high'].astype(float)
    ohlvc['low'] = ohlvc['low'].astype(float)
    ohlvc['close'] = ohlvc['close'].astype(float)
    ohlvc['volume'] = ohlvc['volume'].astype(float)

    ohlvc.to_csv('../data/ohlvc.csv', index=False)

    print(ohlvc.head())
    print(ohlvc.shape)

    return ohlvc

def match_quotes(trades, quotes):

    # Match trades with quotes based on timestamp
    trades = trades.sort_values('timestamp')
    quotes = quotes.sort_values('open_time')

    matched = pd.merge_asof(
        trades,
        quotes,
        left_on='timestamp',
        right_on='open_time',
        direction='backward'
    )

    matched.to_csv('../data/trades_w_quotes.csv', index=False)

    print(matched.head())
    print(matched.shape)

    return matched

def sparsify(df):
    # group trades with the same timestamp and trade sign into a single trades
    # the price is the average is the volume weighted average
    grouped = df.groupby(['timestamp', 'trade_sign']).agg({
        'trade_id': 'first',
        'price': lambda x: np.average(x, weights=df.loc[x.index, 'volume_normalized']),
        'volume': 'sum',
        'volume_normalized': 'sum',
        'log_volume': 'sum',
        'sqrt_volume': 'sum',
        'quote_volume': 'sum'
    }).reset_index()

    grouped.to_csv('../data/trades_sparse.csv', index=False)

    print(grouped.head())
    print(grouped.shape)

    print(f"{len(df)} -> {len(grouped)} trades ({len(df) - len(grouped)} grouped)")
    return grouped


#==================================================================================
# ML Preprocessing
#==================================================================================

def features(trades, windows=[5, 20, 50]):

    features = trades.copy()

    # Basic price features
    features['returns'] = features['price'].pct_change()
    features['log_returns'] = np.log(features['price'] / features['price'].shift(1))

    # Volatility metrics
    for window in windows:
        features[f'volatility_{window}'] = features['returns'].rolling(window).std()
        features[f'price_range_{window}'] = (features['high'] - features['low']).rolling(window).mean()
        features[f'hl_volatility_{window}'] = np.log(features['high'] / features['low']).rolling(window).mean()
    
    # Volume patterns
    features['log_volume'] = np.log(features['volume_x'] + 1e-10)
    features['sqrt_volume'] = np.sqrt(features['volume_x'])

    for window in windows:
        features[f'volume_ma_{window}'] = features['volume_x'].rolling(window).mean()
        features[f'volume_std_{window}'] = features['volume_x'].rolling(window).std()
        features[f'volume_ratio_{window}'] = features['volume_x'] / features[f'volume_ma_{window}']

    # Spread dynamics
    features['relative_spread'] = features['spread'] / features['midpoint']
    features['spread_bps'] = features['spread'] / features['midpoint'] * 10000  # basis points

    for window in windows:
        features[f'spread_ma_{window}'] = features['relative_spread'].rolling(window).mean()
        features[f'spread_std_{window}'] = features['relative_spread'].rolling(window).std()

    # Order flow imbalance
    features['buy_ratio'] = features['taker_buy_volume'] / (features['volume_y'] + 1e-10)
    features['order_imbalance'] = (features['taker_buy_volume'] * 2 - features['volume_y']) / features['volume_y']

    for window in windows:
        features[f'buy_ratio_ma_{window}'] = features['buy_ratio'].rolling(window).mean()
        features[f'imbalance_ma_{window}'] = features['order_imbalance'].rolling(window).mean()

    # Trade sign patterns
    for window in windows:
        features[f'sign_sum_{window}'] = features['trade_sign'].rolling(window).sum()
        features[f'sign_mean_{window}'] = features['trade_sign'].rolling(window).mean()
        features[f'sign_std_{window}'] = features['trade_sign'].rolling(window).std()

    # Trade frequency and timing
    features['trade_count'] = features['count']
    
    for window in windows:
        features[f'trade_count_ma_{window}'] = features['count'].rolling(window).mean()

    # Price impact features (basic)
    features['price_change_1'] = features['price'].shift(-1) - features['price']
    features['price_change_5'] = features['price'].shift(-5) - features['price']
    features['price_change_20'] = features['price'].shift(-20) - features['price']

    # Square-root law baseline
    features['sqrt_impact_baseline'] = features['sqrt_volume'] * features[f'volatility_{windows[1]}']

    # Market regime indicators
    features['price_trend_5'] = (features['price'] - features['price'].shift(5)) / features['price'].shift(5)
    features['price_trend_20'] = (features['price'] - features['price'].shift(20)) / features['price'].shift(20)

    # Drop rows with NaN value
    features = features.dropna()

    features.to_csv('../data/features.csv', index=False)
    print(features.head())
    print(features.shape)

    return features

# utility function to parse cli args 
def parse_args():
    parser = argparse.ArgumentParser(description="Data Processing Script")
    parser.add_argument('--no-collect', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    collect = not args.no_collect
    
    if collect:
        trades = collect_trades()
        ohlvc = collect_quotes()
        trades = sparsify(trades)
        trades_w_quote = match_quotes(trades, ohlvc)
    else:
        trades_w_quote = pd.read_csv('../data/trades_w_quotes.csv')

    feature_df = features(trades_w_quote)
