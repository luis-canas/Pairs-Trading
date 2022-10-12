

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARMA, ARIMA

from scipy.stats import zscore


class Trader:


    def __init__(self,data):

        self.__dict = {'Returns':0}
        self.__models = {'Returns':0}
        self.__all_pairs=[]
        self.__data=data


    def set_pairs(self,pairs):
        
        self.__all_pairs=pairs

    def ARMA_model(self,signal1,signal2):

        spread=signal2-signal1
        normalized_spread=zscore(spread)

        model=ARMA(spread,order=(1,1))
        model.fit(spread)
        return model

    def baseline_model(self,signal1,signal2):

        return 0

    def moving_average(self,signal1, signal2):

        window1=50
        window2=50

        
        # Compute rolling mean and rolling standard deviation
        spread = signal1/signal2
        ma1 = spread.rolling(window=window1,
                                center=False).mean()
        ma2 = spread.rolling(window=window2,
                                center=False).mean()
        std = spread.rolling(window=window2,
                            center=False).std()
        zscore = (ma1 - ma2)/std
        
        # Simulate trading
        # Start with no money and no positions
        returns = 0
        countS1 = 0
        countS2 = 0
        for i in range(len(spread)):
            # Sell short if the z-score is > 1
            if zscore[i] < -1:
                returns += signal1[i] - signal2[i] * spread[i]
                countS1 -= 1
                countS2 += spread[i]
                #print('Selling Ratio %s %s %s %s'%(money, ratios[i], countS1,countS2))
            # Buy long if the z-score is < -1
            elif zscore[i] > 1:
                returns -= signal1[i] - signal2[i] * spread[i]
                countS1 += 1
                countS2 -= spread[i]
                #print('Buying Ratio %s %s %s %s'%(money,ratios[i], countS1,countS2))
            # Clear positions if the z-score between -.5 and .5
            elif abs(zscore[i]) < 0.75:
                returns += signal1[i] * countS1 + signal2[i] * countS2
                countS1 = 0
                countS2 = 0
                #print('Exit pos %s %s %s %s'%(money,ratios[i], countS1,countS2))
                
                
        return returns

    def run_simulation(self,model):

        for signal1,signal2 in self.pairs:
            
            if model == 'MA':

                self.__dict['Returns']+=self.moving_average(signal1,signal2)
                print( self.__dict['Returns'])

            