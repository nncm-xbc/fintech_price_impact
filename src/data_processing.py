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
    trades['timestamp'] = pd.to_datetime(trades['timestamp'], unit='ns')
    quotes['open_time'] = pd.to_datetime(quotes['open_time'], unit='ns')

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

def response_peak(prices, signs, window):
    results = []

    for i in range(len(prices)):
        if i < window:
            results.append(np.nan)
            continue
        
        p = prices[i-window:i].values
        s = signs[i-window:i].values
        
        response_values = []
        for lag in range(1, min(21, len(p))):
            if lag >= len(p):
                break
            price_diffs = p[lag:] - p[:-lag]
            trade_signs = s[:-lag]
            if len(price_diffs) > 0:
                response_values.append(np.mean(price_diffs * trade_signs))
        
        if len(response_values) > 3:
            peak_lag = np.argmax(response_values) + 1
            peak_value = np.max(response_values)
            results.append(peak_lag)
        else:
            results.append(np.nan)
    return pd.Series(results, index=prices.index)

def rolling_gamma(signs, window):
    results = []
    for i in range(len(signs)):
        if i < window:
            results.append(np.nan)
            continue
        
        s = signs[i-window:i].values

        # Sign corr for each lag
        correlations = []
        for lag in range(1, min(21, len(s)//2)):
            if lag >= len(s):
                break
            corr = np.corrcoef(s[:-lag], s[lag:])[0,1]
            if np.isfinite(corr) and corr > 0:
                correlations.append(corr)
        
        if len(correlations) > 3:
            # log(corr) = log(A) - gamma*log(lag)
            lags = np.arange(1, len(correlations) + 1)
            log_corr = np.log(correlations)
            log_lags = np.log(lags)
            gamma = -np.polyfit(log_lags, log_corr, 1)[0]
            results.append(max(0, gamma))
        else:
            results.append(np.nan)

    return pd.Series(results, index=signs.index)

def effective_trades(signs, window):
    results = []
    for i in range(len(signs)):
        if i < window:
            results.append(np.nan)
            continue
        
        s = signs[i-window:i].values
        total_corr = 1.0

        for lag in range(1, min(21, len(s)//2)):
            if lag >= len(s):
                break
            corr = np.corrcoef(s[:-lag], s[lag:])[0,1]
            if np.isfinite(corr) and corr > 0:
                total_corr += corr
            else:
                break 
        
        results.append(total_corr)
    return pd.Series(results, index=signs.index)

def features(trades, windows=[5, 20, 50], min_warmup=100):

    features = trades.copy()
    
    # ================================
    # DATA CLEANING AND PREPARATION
    # ================================
    
    # Clean core columns
    features['price'] = features['price'].ffill().bfill()
    features['midpoint'] = features['midpoint'].ffill().bfill()
    
    volume_col = 'volume_x' if 'volume_x' in features.columns else 'volume_normalized'
    features[volume_col] = features[volume_col].fillna(1)
    
    # Quote/spread data
    if 'midpoint' in features.columns:
        features['midpoint'] = features['midpoint'].ffill().bfill()
    if 'spread' in features.columns:
        features['spread'] = features['spread'].fillna(0)
    if 'high' in features.columns and 'low' in features.columns:
        features['high'] = features['high'].fillna(features['price'])
        features['low'] = features['low'].fillna(features['price'])
    
    # Order flow data
    if 'taker_buy_volume' in features.columns:
        features['taker_buy_volume'] = features['taker_buy_volume'].fillna(0)
    if 'volume_y' in features.columns:
        features['volume_y'] = features['volume_y'].fillna(features[volume_col])
    if 'count' in features.columns:
        features['count'] = features['count'].fillna(1)
    
    # ================================
    # BASIC PRICE FEATURES
    # ================================
    
    features['returns'] = features['price'].pct_change().fillna(0)
    features['log_returns'] = np.log(features['price'] / features['price'].shift(1)).fillna(0)

    # ================================
    # VOLATILITY METRICS
    # ================================
    
    for window in windows:
        # Price volatility - expanding window fallback for early observations
        vol_rolling = features['returns'].rolling(window).std()
        vol_expanding = features['returns'].expanding().std()
        features[f'volatility_{window}'] = vol_rolling.fillna(vol_expanding).fillna(0)
        
        # Price range features
        if 'high' in features.columns and 'low' in features.columns:
            price_range = features['high'] - features['low']
            range_rolling = price_range.rolling(window).mean()
            range_expanding = price_range.expanding().mean()
            features[f'price_range_{window}'] = range_rolling.fillna(range_expanding).fillna(0)
            
            # High-low volatility
            hl_ratio = np.log((features['high'] + 1e-10) / (features['low'] + 1e-10))
            hl_rolling = hl_ratio.rolling(window).mean()
            hl_expanding = hl_ratio.expanding().mean()
            features[f'hl_volatility_{window}'] = hl_rolling.fillna(hl_expanding).fillna(0)
        else:
            # Fallback if no high/low data
            features[f'price_range_{window}'] = features[f'volatility_{window}']
            features[f'hl_volatility_{window}'] = features[f'volatility_{window}']
    
    # ================================
    # VOLUME PATTERNS
    # ================================
    
    features['log_volume'] = np.log(features[volume_col] + 1e-10)
    features['sqrt_volume'] = np.sqrt(features[volume_col])

    for window in windows:
        # Volume moving averages
        vol_ma_rolling = features[volume_col].rolling(window).mean()
        vol_ma_expanding = features[volume_col].expanding().mean()
        features[f'volume_ma_{window}'] = vol_ma_rolling.fillna(vol_ma_expanding).fillna(1)
        
        # Volume standard deviation
        vol_std_rolling = features[volume_col].rolling(window).std()
        vol_std_expanding = features[volume_col].expanding().std()
        features[f'volume_std_{window}'] = vol_std_rolling.fillna(vol_std_expanding).fillna(0)
        
        # Volume ratio
        features[f'volume_ratio_{window}'] = features[volume_col] / (features[f'volume_ma_{window}'] + 1e-10)

    # ================================
    # SPREAD DYNAMICS
    # ================================
    
    if 'spread' in features.columns and 'midpoint' in features.columns:
        features['relative_spread'] = features['spread'] / (features['midpoint'] + 1e-10)
        features['spread_bps'] = features['relative_spread'] * 10000  # basis points

        for window in windows:
            spread_ma_rolling = features['relative_spread'].rolling(window).mean()
            spread_ma_expanding = features['relative_spread'].expanding().mean()
            features[f'spread_ma_{window}'] = spread_ma_rolling.fillna(spread_ma_expanding).fillna(0.0001)
            
            spread_std_rolling = features['relative_spread'].rolling(window).std()
            spread_std_expanding = features['relative_spread'].expanding().std()
            features[f'spread_std_{window}'] = spread_std_rolling.fillna(spread_std_expanding).fillna(0)
    else:
        # Create dummy spread features if not available
        features['relative_spread'] = 0.0001  # Small constant spread
        features['spread_bps'] = 1.0
        for window in windows:
            features[f'spread_ma_{window}'] = 0.0001
            features[f'spread_std_{window}'] = 0

    # ================================
    # ORDER FLOW IMBALANCE
    # ================================
    
    if 'taker_buy_volume' in features.columns and 'volume_y' in features.columns:
        features['buy_ratio'] = features['taker_buy_volume'] / (features['volume_y'] + 1e-10)
        features['order_imbalance'] = (features['taker_buy_volume'] * 2 - features['volume_y']) / (features['volume_y'] + 1e-10)

        for window in windows:
            buy_ma_rolling = features['buy_ratio'].rolling(window).mean()
            buy_ma_expanding = features['buy_ratio'].expanding().mean()
            features[f'buy_ratio_ma_{window}'] = buy_ma_rolling.fillna(buy_ma_expanding).fillna(0.5)
            
            imbal_ma_rolling = features['order_imbalance'].rolling(window).mean()
            imbal_ma_expanding = features['order_imbalance'].expanding().mean()
            features[f'imbalance_ma_{window}'] = imbal_ma_rolling.fillna(imbal_ma_expanding).fillna(0)
    else:
        # Create neutral order flow features if data is not available
        features['buy_ratio'] = 0.5
        features['order_imbalance'] = 0
        for window in windows:
            features[f'buy_ratio_ma_{window}'] = 0.5
            features[f'imbalance_ma_{window}'] = 0

    # ================================
    # TRADE SIGN PATTERNS
    # ================================
    
    for window in windows:
        # Sign sum
        sign_sum_rolling = features['trade_sign'].rolling(window).sum()
        sign_sum_expanding = features['trade_sign'].expanding().sum()
        features[f'sign_sum_{window}'] = sign_sum_rolling.fillna(sign_sum_expanding).fillna(0)
        
        # Sign mean
        sign_mean_rolling = features['trade_sign'].rolling(window).mean()
        sign_mean_expanding = features['trade_sign'].expanding().mean()
        features[f'sign_mean_{window}'] = sign_mean_rolling.fillna(sign_mean_expanding).fillna(0)
        
        # Sign std
        sign_std_rolling = features['trade_sign'].rolling(window).std()
        sign_std_expanding = features['trade_sign'].expanding().std()
        features[f'sign_std_{window}'] = sign_std_rolling.fillna(sign_std_expanding).fillna(0.5)

    # ================================
    # TRADE FREQUENCY AND TIMING
    # ================================
    
    if 'count' in features.columns:
        features['trade_count'] = features['count']
        for window in windows:
            count_ma_rolling = features['count'].rolling(window).mean()
            count_ma_expanding = features['count'].expanding().mean()
            features[f'trade_count_ma_{window}'] = count_ma_rolling.fillna(count_ma_expanding).fillna(1)
    else:
        features['trade_count'] = 1
        for window in windows:
            features[f'trade_count_ma_{window}'] = 1

    # ================================
    # PRICE IMPACT FEATURES
    # ================================
    
    price_change_1 = features['price'].shift(-1) - features['price']
    price_change_5 = features['price'].shift(-5) - features['price']
    price_change_20 = features['price'].shift(-20) - features['price']
    
    # Fill forward-looking NaN values with mean reversion assumption
    features['price_change_1'] = price_change_1.fillna(price_change_1.mean())
    features['price_change_5'] = price_change_5.fillna(price_change_5.mean())
    features['price_change_20'] = price_change_20.fillna(price_change_20.mean())

    # ================================
    # SQUARE-ROOT LAW BASELINE
    # ================================
    
    features['sqrt_impact_baseline'] = features['sqrt_volume'] * features[f'volatility_{windows[1]}']

    # ================================
    # MARKET REGIME INDICATORS (PRICE TRENDS)
    # ================================
    
    price_5_ago = features['price'].shift(5).fillna(features['price'])
    price_20_ago = features['price'].shift(20).fillna(features['price'])
    
    features['price_trend_5'] = (features['price'] - price_5_ago) / (price_5_ago + 1e-10)
    features['price_trend_20'] = (features['price'] - price_20_ago) / (price_20_ago + 1e-10)

    # ================================
    # COMPLEX FEATURES
    # ================================
    
    # Rolling correlation between returns and lagged trade signs
    response_proxy_list = []
    for i in range(len(features)):
        if i < windows[0]:
            response_proxy_list.append(windows[0] // 2)  # Default to mid-window
        else:
            window_returns = features['returns'].iloc[i-windows[0]:i]
            window_signs = features['trade_sign'].iloc[i-windows[0]:i]
            if len(window_returns) > 2 and window_returns.std() > 0 and window_signs.std() > 0:
                try:
                    corr = np.corrcoef(window_returns, window_signs)[0,1]
                    response_proxy_list.append(abs(corr) * 10 + 1)  # Convert to lag-like value
                except:
                    response_proxy_list.append(windows[0] // 2)
            else:
                response_proxy_list.append(windows[0] // 2)
    
    features['response_peak_lag'] = response_proxy_list

    sign_autocorr_list = []  # Autocorrelation
    for i in range(len(features)):
        if i < windows[0]:
            sign_autocorr_list.append(0.5)  # Neutral value
        else:
            window_signs = features['trade_sign'].iloc[i-windows[0]:i]
            if len(window_signs) > 2 and window_signs.std() > 0:
                try:
                    corr = np.corrcoef(window_signs[:-1], window_signs[1:])[0,1]
                    sign_autocorr_list.append(max(0, min(1, corr)))  # Clip to [0,1]
                except:
                    sign_autocorr_list.append(0.5)
            else:
                sign_autocorr_list.append(0.5)
    
    features['corr_gamma'] = sign_autocorr_list

    effective_trades_list = []  # Trade consistency
    for i in range(len(features)):
        if i < windows[0]:
            effective_trades_list.append(1)
        else:
            window_signs = features['trade_sign'].iloc[i-windows[0]:i]
            if len(window_signs) > 0:
                consistency = 1 + abs(window_signs.mean()) * len(window_signs)
                effective_trades_list.append(consistency)
            else:
                effective_trades_list.append(1)
    
    features['effective_trades'] = effective_trades_list

    # ================================
    # LOCAL DIFFUSION VS RESPONSE
    # ================================
    
    # Local diffusion - variance of returns
    diffusion_rolling = features['returns'].rolling(windows[0]).var()
    diffusion_expanding = features['returns'].expanding().var()
    features['local_diffusion'] = diffusion_rolling.fillna(diffusion_expanding).fillna(0)
    
    # Simplified local response strength
    momentum_rolling = features['returns'].rolling(windows[0]).mean()
    momentum_expanding = features['returns'].expanding().mean()
    local_momentum = momentum_rolling.fillna(momentum_expanding).fillna(0)
    
    trade_intensity_rolling = features['trade_sign'].rolling(windows[0]).std()
    trade_intensity_expanding = features['trade_sign'].expanding().std()
    local_trade_intensity = trade_intensity_rolling.fillna(trade_intensity_expanding).fillna(0.5)
    
    features['local_response_strength'] = np.abs(local_momentum * local_trade_intensity)

    # ================================
    # CRITICAL BEHAVIOR INDICATOR
    # ================================
    
    # Distance from criticality
    features['criticality'] = np.abs(2 * ((1 - features['corr_gamma']) / 2) + features['corr_gamma'] - 1)

    # ================================
    # WARM-UP PERIOD
    # ================================
    
    # Apply warm-up period to ensure all features are stable
    features = features.iloc[min_warmup:].copy()
    
    # Safety checks
    features = features.replace([np.inf, -np.inf], 0)
    features = features.fillna(0)
    
    # Check no NaN values remain
    nan_counts = features.isnull().sum()
    if nan_counts.sum() > 0:
        print("Warning: NaN values remain in columns:")
        for col in nan_counts[nan_counts > 0].index:
            print(f"  {col}: {nan_counts[col]} NaN values")
            features[col] = features[col].fillna(0)
    
    print(f"Features created: {features.shape}")
    print(f"Feature columns: {len(features.columns)}")
    print(f"No NaN values: {not features.isnull().any().any()}")
    
    return features

def feature_columns(df, exclude_targets=True):

    exclude_cols = [
        'timestamp', 'trade_id', 'open_time', 'ignore',
        'price', 'open', 'high', 'low', 'close', 'close_time'  # raw price data excluded
    ]
    
    if exclude_targets:
        exclude_cols.extend([
            'price_change_1', 'price_change_5', 'price_change_20'  # target variables
        ])
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols

def target(features_df, horizon=5):

    # Ensure price column exists
    if 'price' not in features_df.columns:
        raise ValueError("Price column not found in features DataFrame")
    
    price = features_df['price'].fillna(method='ffill').fillna(method='bfill')
    trade_sign = features_df['trade_sign'].fillna(0)
    
    # Future price change
    price_change = price.shift(-horizon) - price
    
    # handle end-of-series NaN values
    price_change = price_change.fillna(price_change.mean())
    
    # Signed impact target
    target = price_change * trade_sign
    
    return target.fillna(0)

# utility function to parse cli args 
def parse_args():
    parser = argparse.ArgumentParser(description="Data Processing Script")
    parser.add_argument('--no-collect', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    collect = not args.no_collect
    
    if collect:
        # trades = collect_trades()
        # ohlvc = collect_quotes()
        # trades = sparsify(trades)
        trades = pd.read_csv('../data/trades_sparse.csv')  
        ohlvc = pd.read_csv('../data/ohlvc.csv')  
        trades_w_quote = match_quotes(trades, ohlvc)
    else:
        trades_w_quote = pd.read_csv('../data/trades_w_quotes.csv')

    feature_df = features(trades_w_quote)
    feature_cols = feature_columns(feature_df)

    target = target(feature_df)

    feature_df.to_csv("../data/features_test.csv", index=False)
    target.to_csv("../data/target_test.csv", index=False, header=['price_change_5'])

    print(f"Final dataset: {len(feature_df)} samples, {len(feature_cols)} features")
    print(f"Feature columns: {feature_cols}")
