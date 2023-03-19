
import pprint
from utils import *


class Portfolio:


    def __init__(self,data,index,sector,start_date,end_date,months_trading,months_forming,pairs_alg,trading_alg):

        self.data=data
        self.index=index
        self.sector=sector
        self.tickers=data.keys()
        self.start_date=date_string(start_date)
        self.end_date=date_string(end_date)
        self.months_trading=months_trading
        self.months_forming=months_forming
        self.pairs_alg=pairs_alg
        self.trading_alg=trading_alg



        self.pair_info=[]
        self.portfolio_info=[]


    def report(self,pairs,performance,verbose=False):

        self.pair_info.append(pairs)
        self.portfolio_info.append(performance)

        if verbose:
            pprint.pprint([pairs,performance],depth=2,sort_dicts=False)

    def evaluate(self, verbose=False):
        
        if verbose:
            for simul in range(len(self.portfolio_info)):
                pprint.pprint([self.pair_info[simul],self.portfolio_info[simul]],depth=2,sort_dicts=False)




