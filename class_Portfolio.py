

from pandas_datareader import data
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import pandas as pd
from datetime import datetime
import yfinance as yf

import warnings
warnings.filterwarnings("ignore")

class Portfolio:


    def __init__(self):
        self.__train_val_split=1
        self.__data_train=0
        self.__data_val=0
        self.__data=[]

    def get_train_val(self):
        return self.__data_train, self.__data_val

    def portfolio1(self):

        start = datetime(2013, 1, 1)
        end = datetime(2018, 1, 1)
        tickers = ['AAPL', 'ADBE', 'ORCL', 'EBAY', 'MSFT', 'QCOM', 'HPQ', 'JNPR', 'AMD', 'IBM', 'SPY']

        self.__data=yf.download(tickers, start, end)['Close']
        split=int(len(self.__data)*self.__train_val_split)

        self.__data_train=self.__data[:split]
        self.__data_val=self.__data[split:]

        


    def portfolio2(self):


        start = datetime(2013, 1, 1)
        end = datetime(2015, 1, 1)
        df = pd.read_csv('nasdaq_screener.csv')

        mask=df['Sector'].str.contains('Finance')
        mask=mask.where(pd.notnull(mask), False).tolist()
        tickers=df['Symbol']
        tickers=tickers[mask].tolist()

        self.__data= data.get_data_yahoo(tickers,start=start,end=end)['Close'].dropna()
        split=int(len(self.data)*self.__train_val_split)

        self.__data_train=self.__data[:split]
        self.__data_val=self.__data[split:]
 
