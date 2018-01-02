from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

DATA_URL = 'http://192.168.1.46:9900/'


def step_signal(start_price, period_1=50, volatility=0.05):
    data = []
    delta = 1
    start_date = datetime.now() - timedelta(days=1)
    for i in range(360 * 24):
        price = start_price + start_price * delta * volatility
        if i and i % period_1 == 0:
            delta *= -1
        date = start_date + timedelta(seconds=(i + 1) * 10)
        data.append({'price': price, 'datetime': date, 'serial_number': i, 'symbol': 'BTCUSD'})
    return pd.DataFrame(data)


def wavy_signal(start_price, period_1=50, period_2=100, epsilon=0.1, volatility=0.05):
    data = []
    start_date = datetime.now() - timedelta(days=1)
    for i in range(360 * 24):
        delta = (1 - epsilon) * np.sin(2 * i * np.pi / period_1) + epsilon * np.sin(2 * i * np.pi / period_2)
        price = start_price + start_price * delta * volatility
        date = start_date + timedelta(seconds=(i + 1) * 10)
        data.append({'price': price, 'datetime': date, 'serial_number': i, 'symbol': 'BTCUSD'})
    return pd.DataFrame(data)


def historical(symbol, _type='tickers', period='h'):
    data = []
    json = requests.get(DATA_URL + _type, params={'symbols': symbol, 'since': '1' + period}).json()
    for i in json['data']:
        d = i.copy()
        del d['time']
        d['datetime'] = datetime.fromtimestamp(i['time'])
        if _type == 'tickers':
            del d['last_price']
            d['price'] = i['last_price']
        elif _type == 'candles':
            d['price'] = i['close']
        data.append(d)
    return pd.DataFrame(data)
