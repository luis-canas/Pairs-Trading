


from datetime import datetime
import yfinance as yf

class Portfolio:


    def __init__(self):
        self.train_val_split=1
        self.data_train=0
        self.data_val=0
        self.data=0

    def get_train_val(self):
        return self.data_train, self.data_val

    def portfolio1(self):

        start = datetime(2013, 1, 1)
        end = datetime(2018, 1, 1)
        tickers = ['AAPL', 'ADBE', 'ORCL', 'EBAY', 'MSFT', 'QCOM', 'HPQ', 'JNPR', 'AMD', 'IBM', 'SPY']

        self.data=yf.download(tickers, start, end)['Close']
        split=int(len(self.data)*self.train_val_split)

        self.data_train=self.data[:split]
        self.data_val=self.data[split:]
 
