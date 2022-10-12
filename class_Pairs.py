

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.api import OLS
from statsmodels.tsa.stattools import coint, adfuller
from hurst import compute_Hc as hurst_exponent
from scipy.stats import zscore

from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler




class Pairs:

    """
    A class used to represent trading Pairs

    """
    __PLOT=False

    def __init__(self, data):

        
        self.__all_pairs = []
        self.__data = data
        self.__tickers = data.keys()
        self.__start = data.index[0]._date_repr
        self.__end = data.index[-1]._date_repr

    def __is_stationary(self,signal, threshold):

        signal=np.asfarray(signal)
        return True if adfuller(signal)[1] < threshold else False


    def __cointegrated_pairs(self):

        data = self.__data
        tickers = self.__tickers
        n_pairs = len(tickers)

        pvalue_threshold = 0.05
        hurst_threshold = 0.5  # mean reversing threshold

        pairs = []

        for i in range(n_pairs):

            signal1 = data[tickers[i]]


            for j in range(i+1, n_pairs):

                signal2 = data[tickers[j]]
                
                if self.__Engle_Granger(signal1, signal2, pvalue_threshold, hurst_threshold):
                    pairs.append((tickers[i], tickers[j]))

        self.__all_pairs = pairs

    def __Engle_Granger(self, signal1, signal2, pvalue_threshold, hurst_threshold):

        beta = OLS(signal2, signal1).fit().params[0]
        spread = signal2-beta*signal1
        result = coint(signal1, signal2)
        score = result[0]
        pvalue = result[1]
        hurst, _, _ = hurst_exponent(spread)
  
        if(self.__PLOT and pvalue <= pvalue_threshold and hurst <= hurst_threshold):
            plt.figure(figsize=(12, 6))
            normalized_spread = zscore(spread)
            normalized_spread.plot()
            standard_devitation = np.std(normalized_spread)
            plt.axhline(zscore(spread).mean())
            plt.axhline(standard_devitation, color='green')
            plt.axhline(-standard_devitation, color='green')
            plt.axhline(2*standard_devitation, color='red')
            plt.axhline(-2*standard_devitation, color='red')
            plt.xlim(self.__start,
                     self.__end)
            plt.show()

        return True if pvalue <= pvalue_threshold and hurst <= hurst_threshold else False

    def find_pairs(self,model,verbose=False):

        function = {'COINT':self.__cointegrated_pairs}
        function[model]()
        
        if verbose:   
            print("\n************************************************\n",
                    "\nModel: ",model,
                    "\nNumber of pairs: ", len( self.__all_pairs),
                    "\nNumber of unique elements: ",len( np.unique(self.__all_pairs)),
                    "\nPairs: ",self.__all_pairs,
                    "\n\n************************************************\n")
                    
        return self.__all_pairs

    # def __cointegrated_pairs(self):

    #     data = self.__data
    #     tickers = self.__tickers
    #     n_pairs = len(tickers)

    #     adfuller_threshold = 0.1
    #     pvalue_threshold = 0.05
    #     hurst_threshold = 0.5  # mean reversing threshold

    #     pairs = []

    #     for i in range(n_pairs):

    #         signal1 = data[tickers[i]]

    #         if self.__is_stationary(signal1, adfuller_threshold):

    #             for j in range(i+1, n_pairs):

    #                 signal2 = data[tickers[j]]

    #                 if self.__is_stationary(signal2, adfuller_threshold):
                
    #                     beta = OLS(signal2, signal1).fit().params[0]
    #                     spread = signal2-beta*signal1

    #                     if self.__is_stationary(spread, adfuller_threshold):

    #                         hurst, _, _ = hurst_exponent(spread)

                            
    #                         if hurst<hurst_threshold:
    #                             pairs.append((tickers[i], tickers[j]))

    #     self.__all_pairs = pairs