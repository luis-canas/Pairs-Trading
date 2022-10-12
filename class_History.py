

from pandas_datareader import data
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import pandas as pd
from datetime import datetime
import yfinance as yf

import warnings
warnings.filterwarnings("ignore")

class History:


    def __init__(self):
        pass

    def get_data(self,history,start,end):

        function = {'DEFAULT':self.__default}

        
        return function[history](*start,*end)

    def __default(self,start_y,start_m,start_d,end_y,end_m,end_d):


        start = datetime(start_y,start_m,start_d)
        end = datetime(end_y,end_m,end_d)
        tickers = ['AAPL', 'ADBE', 'ORCL', 'EBAY', 'MSFT', 'QCOM', 'HPQ', 'JNPR', 'AMD', 'IBM', 'SPY']

        return yf.download(tickers, start, end)['Close']
        


    def __portfolio(self,start_y,start_m,start_d,end_y,end_m,end_d):


        start = datetime(start_y,start_m,start_d)
        end = datetime(end_y,end_m,end_d)
        df = pd.read_csv('nasdaq_screener.csv')

        mask=df['Sector'].str.contains('Finance')
        mask=mask.where(pd.notnull(mask), False).tolist()
        tickers=df['Symbol']
        tickers=tickers[mask].tolist()


        self.__data= data.get_data_yahoo(tickers,start=start,end=end)['Close']
