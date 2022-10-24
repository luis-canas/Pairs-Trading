

from pandas_datareader import data
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import pandas as pd
from datetime import datetime
import yfinance as yf
import pickle

import warnings
warnings.filterwarnings("ignore")

class History:


    def __init__(self):
        self.__start=[]
        self.__end=[]

    def set_date(self,start_y,start_m,start_d,end_y,end_m,end_d):

        self.__start = datetime(start_y,start_m,start_d)
        self.__end = datetime(end_y,end_m,end_d)


    def get_data(self,history):

        function = {'DEFAULT':self.__default,'PICKLE':self.__pickle}

        
        return function[history]()

    def __default(self):


        start = self.__start
        end = self.__end

        tickers = ['AAPL', 'ADBE', 'ORCL', 'EBAY', 'MSFT', 'QCOM', 'HPQ', 'JNPR', 'AMD', 'IBM', 'SPY']

        return yf.download(tickers, start, end)['Close']
        


    def __portfolio(self):


        start = self.__start
        end = self.__end

        df = pd.read_csv('nasdaq_screener.csv')

        mask=df['Sector'].str.contains('Finance')
        mask=mask.where(pd.notnull(mask), False).tolist()
        tickers=df['Symbol']
        tickers=tickers[mask].tolist()


        data= data.get_data_yahoo(tickers,start=start,end=end)['Close']

        return data

    def __pickle(self):

        return pd.read_pickle('commodity_ETFs_from_2014_complete.pickle')
