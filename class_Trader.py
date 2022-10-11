

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARMA, ARIMA

from scipy.stats import zscore


class Trader:


    def __init__(self,data):

        self.all_pairs=[]
        self.data=data

    def set_pairs(self,pairs):
        
        self.all_pairs=pairs

    def ARMA_model(self,signal1,signal2):

        spread=signal2-signal1
        normalized_spread=zscore(spread)

        model=ARMA(spread,order=(1,1))
        model.fit(spread)
        return model

    def baseline_model(self,signal1,signal2):

        spread=signal2-signal1
        normalized_spread=zscore(spread)

        model=ARMA(spread,order=(1,1))
        model.fit(spread)
        return 0

    def run_simulation(self):

        for signal1,signal2 in self.pairs:
            self.ARMA_model(signal1,signal2)