import requests
import pandas as pd
import arrow
import datetime
import numpy as np

def get_quote_data(symbol='AAPL', data_range='7d', data_interval='1s'):
    res = requests.get(f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={data_interval}')
    data = res.json()
    body = data['chart']['result'][0]
    dt = datetime.datetime
    dt = pd.Series(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None), body['timestamp']), name='dt')
    df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
    dg = pd.DataFrame(body['timestamp'])
    df = df.loc[:, ('close', 'volume')]
    df.dropna(inplace=True)  # removing NaN rows
    df.columns = ['CLOSE', 'VOLUME']  # Renaming columns in pandas
    df.to_csv('out.csv')

    return df

def get_enhanced_quote_data(symbol='AAPL', data_range='7d', data_interval='1m'):
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={data_interval}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        return None
        
    data = res.json()
    body = data['chart']['result'][0]
    
    dt = pd.Series(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None), 
                      body['timestamp']), name='dt')
    
    # Get all OHLCV data
    quotes = body['indicators']['quote'][0]
    df = pd.DataFrame(quotes, index=dt)
    
    # Select all relevant fields
    df = df.loc[:, ('open', 'high', 'low', 'close', 'volume')]
    df.dropna(inplace=True)
    df.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    
    # Add microstructure features
    df['SPREAD'] = df['HIGH'] - df['LOW']
    df['MIDPOINT'] = (df['HIGH'] + df['LOW']) / 2
    df['RETURNS'] = df['CLOSE'].pct_change()
    df['DOLLAR_VOLUME'] = df['CLOSE'] * df['VOLUME']
    df['HOUR'] = df.index.hour
    df['MINUTE'] = df.index.minute

    df.to_csv('aapl_out.csv')
    
    return df

data = get_enhanced_quote_data('AAPL', '7d', '1m')
print(data)