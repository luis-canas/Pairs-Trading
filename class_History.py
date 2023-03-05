

# from pandas_datareader import data
# from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import pandas as pd
from datetime import datetime
import yfinance as yf
import pickle
from os.path import isfile

from utils import date_string
import warnings
warnings.filterwarnings("ignore")

class History:


    def __init__(self):
        self.__start=[]
        self.__end=[]

    def set_date(self,start,end):

        self.__start = start
        self.__end = end


    def get_data(self,history):

        function = {'DEFAULT':self.__default,'PORTFOLIO':self.__portfolio,'PICKLE':self.__pickle}

        
        return function[history]()

    def __default(self):


        start = datetime(*self.__start)
        end = datetime(*self.__end)

        tickers = ['AAPL', 'ADBE', 'ORCL', 'EBAY', 'MSFT', 'QCOM', 'HPQ', 'JNPR', 'AMD', 'IBM', 'SPY']

        return yf.download(tickers, start, end)['Close']
        


    def __portfolio(self):
        index='s&p500'
        sector='Financials'

        start = self.__start
        end = self.__end

        if not isfile(f'data/{index}_{sector}_{date_string(start)}_{date_string(end)}.csv'):
            df = pd.read_csv(f'data/{index}_screener.csv',encoding='latin1')

            mask=df['Sector'].str.contains(sector)
            mask=mask.where(pd.notnull(mask), False).tolist()
            tickers=df['Symbol']
            tickers=tickers[mask].tolist()

            data=yf.download(tickers,start=datetime(*start),end=datetime(*end))['Close']
            
            nan_value = float("NaN")
            data.replace("", nan_value, inplace=True)
            data.dropna(how='all', axis=1, inplace=True)
            data.to_csv(f'data/{index}_{sector}_{date_string(start)}_{date_string(end)}.csv')
        else:
            data = pd.read_csv(f'data/{index}_{sector}_{date_string(start)}_{date_string(end)}.csv',index_col='Date')
        return data

    def __pickle(self):      

        return pd.read_pickle('commodity_ETFs_interpolated_screened.pickle')
