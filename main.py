from class_Pairs import Pairs
from class_Trader import Trader

import numpy as np

from datetime import datetime

start = datetime(2013, 1, 1)
end = datetime(2018, 1, 1)
tickers = ['AAPL', 'ADBE', 'ORCL', 'EBAY', 'MSFT', 'QCOM', 'HPQ', 'JNPR', 'AMD', 'IBM', 'SPY']


# Test Tickers
cointegration=Pairs(start,end,tickers)
selected_pairs=cointegration.get_pairs()
print(selected_pairs)
