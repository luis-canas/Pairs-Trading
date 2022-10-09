
from tracemalloc import start
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.api import OLS
from statsmodels.tsa.stattools import coint,adfuller
from hurst import compute_Hc as hurst_exponent
from scipy.stats import zscore

import seaborn

import yfinance as yf


SHOW_GRAPH=True


class Pairs:

    
    def __init__(self,start,end,tickers):

        self.pairs=[]
        self.start=start
        self.end=end
        self.pvalue_threshold=0.05
        self.hurst_threshold=0.5 #mean reversing threshold
        self.tickers=tickers
        self.data = yf.download(tickers, start, end)['Close']
        

        self.cointegrated_pairs()

    def get_pairs(self):
        return self.pairs

    def stationarity(self,signal):
        signal = np.ravel(signal)
        if adfuller(signal)[1] < self.pvalue_threshold:
            return True
        else:
            return False

    def cointegrated_pairs(self):

        data=self.data
        n_pairs = data.shape[1]
        score_matrix = np.zeros((n_pairs, n_pairs))
        pvalue_matrix = np.ones((n_pairs, n_pairs))
        keys = data.keys()
        pairs = []

        for i in range(n_pairs):
            for j in range(i+1, n_pairs):
                
                pair1 = data[keys[i]]
                pair2 = data[keys[j]]

                if (not self.stationarity(pair1) and not self.stationarity(pair2)):

                    
                    score, pvalue,hurst=self.Engle_Granger(pair1,pair2)

                    score_matrix[i, j] = score
                    pvalue_matrix[i, j] = pvalue
                    
                    if pvalue <= self.pvalue_threshold and hurst <= self.hurst_threshold:
                        pairs.append((keys[i], keys[j]))

        if(SHOW_GRAPH):
            fig, ax = plt.subplots(figsize=(10,10))
            seaborn.heatmap(pvalue_matrix, xticklabels=self.tickers, yticklabels=self.tickers , mask = (pvalue_matrix >= self.pvalue_threshold))
            plt.show()

        self.pairs=pairs



    def Engle_Granger(self,signal1,signal2):
        #If xt and yt are non-stationary and order of integration d=1, then a linear combination of them must be stationary for some value of 
        # beta  and ut. In other words:
        #yt-beta*xt=ut
        beta=OLS(signal2, signal1).fit().params[0]
        spread=signal2-beta*signal1
        result=coint(signal1, signal2)
        score=result[0]
        pvalue=result[1]
        hurst,_,_=hurst_exponent(spread)

        if(SHOW_GRAPH and pvalue <= self.pvalue_threshold and hurst <= self.hurst_threshold):
            normalized_spread=zscore(spread)
            normalized_spread.plot(figsize=(12,6))
            standard_devitation=np.std(normalized_spread)
            plt.axhline(zscore(spread).mean())
            plt.axhline(standard_devitation, color='green')
            plt.axhline(-standard_devitation, color='green')
            plt.axhline(2*standard_devitation, color='red')
            plt.axhline(-2*standard_devitation, color='red')
            plt.xlim(self.start.strftime("%m-%d-%Y"), self.end.strftime("%m-%d-%Y"))
            plt.show()
            
        return score,pvalue,hurst
        





