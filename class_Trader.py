

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.api import OLS
from statsmodels.tsa.arima_model import ARMA, ARIMA

from scipy.stats import zscore

class Trader:

    __PLOT=False

    def __init__(self,data):

        self.__all_pairs=[]
        self.__data=data

    def set_pairs(self,pairs):
        
        self.__all_pairs=pairs

    def __ARMA_model(self,signal1,signal2):
        return 

    def __baseline_model(self,signal1,signal2):
        return 0

    def __moving_average(self,signal1, signal2):

        "https://www.quantrocket.com/codeload/quant-finance-lectures/quant_finance_lectures/Lecture42-Introduction-to-Pairs-Trading.ipynb.html"

        window = 30
        rolling_beta = [np.nan] * window
        for n in range(window, len(signal1)):
            y = signal1[(n - window):n]
            x = signal2[(n - window):n]
            rolling_beta.append(OLS(y, x).fit().params[0])

        rolling_beta = pd.Series(rolling_beta, index=signal2.index)

        spread = signal2 - rolling_beta * signal1
        spread.name = 'spread'

        # Get the 1 day moving average of the price spread
        spread_mavg1 = spread.rolling(window=1).mean()
        spread_mavg1.name = 'spread 1d mavg'

        # Get the 30 day moving average
        spread_mavg30 = spread.rolling(30).mean()
        spread_mavg30.name = 'spread 30d mavg'


                
        # Take a rolling 30 day standard deviation
        std_30 = spread.rolling(window).std()
        std_30.name = 'std 30d'

        # Compute the z score for each day
        zscore_30_1 = (spread_mavg1 - spread_mavg30)/std_30
        zscore_30_1.name = 'z-score'

        returns=[]
        open_position=False
        entry=0

        for i in range(window,len(spread)):
            if open_position:
                if zscore_30_1[i]>3.0 or  zscore_30_1[i]<-3.0:
                    open_position=False
                    returns.append(-abs(spread[i]-entry))
                elif zscore_30_1[i]>0 and direction or zscore_30_1[i] < 0 and not direction:   
                    open_position=False             
                    returns.append(abs(spread[i]-entry))
                else:
                    returns.append(0)
            else:
                if zscore_30_1[i]>1.0 or  zscore_30_1[i]<-1.0:
                    open_position=True
                    entry=spread[i]
                    direction=False if  zscore_30_1[i]>1.0 else True
                    
                returns.append(0)

        if(self.__PLOT):

            plt.plot(spread_mavg1.index, spread_mavg1.values)
            plt.plot(spread_mavg30.index, spread_mavg30.values)
            plt.legend(['1 Day Spread MAVG', '30 Day Spread MAVG'])
            plt.ylabel('Spread')
            plt.show()

            plt.plot(zscore_30_1.index, zscore_30_1.values)
            plt.axhline(0, color='black')
            plt.axhline(1.0, color='red', linestyle='--')
            plt.show()

        return sum(returns)

    def run_simulation(self,model,verbose=False):

        function = {'MA':self.__moving_average}
        summary={'Returns':0}

        if verbose:
            print("\n************************************************\n",
                    "\nModel: ",model)

        for signal1,signal2 in self.__all_pairs:
            

            returns=function[model](self.__data[signal1],self.__data[signal2])
            summary['Returns']+=returns

            if verbose:
                print("Pair ({}-{}) returns: {}".format(signal1,signal2,returns))

                


        
        print("Portfolio returns: ",summary['Returns'],
                "\n\n************************************************\n")