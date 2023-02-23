

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.api import OLS
from statsmodels.tsa.arima_model import ARMA, ARIMA

from scipy.stats import zscore

class Trader:

    __PLOT=False

    def __init__(self,data,):

        self.__all_pairs=[]
        self.__data=data

    def set_pairs(self,pairs):
        
        self.__all_pairs=pairs

    def __ARIMA_model(self,signal1,signal2):
        return 

    def __baseline_model(self,signal1,signal2):
        return 0

    # def __threshold(self,signal1, signal2,stop_loss=4,entry=2,close=0):


    #     beta = OLS(signal2, signal1).fit().params[0]
    #     spread = signal2-beta*signal1


    #     # Compute the z score for each day
    #     zs = zscore(spread)

    #     returns=[]
    #     open_position=False
    #     initial_value=0
    #     p=0
    #     l=0
    #     for i in range(len(spread)):
    #         if open_position:
    #             if zs[i]>stop_loss or  zs[i]<-stop_loss:
    #                 open_position=False
    #                 returns.append(-abs(spread[i]-initial_value))
    #                 l+=1
    #             elif zs[i]>close and rising or zs[i] < close and not rising:   
    #                 open_position=False             
    #                 returns.append(abs(spread[i]-initial_value))
    #                 p+=1
    #             else:
    #                 returns.append(0)
    #         else:
    #             if zs[i]>entry or  zs[i]<-entry:
    #                 open_position=True
    #                 initial_value=spread[i]
    #                 rising=False if  zs[i]>2.0 else True
                    
    #             returns.append(0)

    #     print('profit positions=',p)
    #     print('stop loss positions=',l)
    #     if(self.__PLOT):

    #         plt.plot(zs.index, zs.values)
    #         plt.plot(zs.index, zs.values)
    #         plt.legend(['1 Day Spread MAVG', '30 Day Spread MAVG'])
    #         plt.ylabel('Spread')
    #         plt.show()

    #         plt.plot(zs.index, zs.values)
    #         plt.axhline(0, color='black')
    #         plt.axhline(1.0, color='blue', linestyle='--')
    #         plt.axhline(2.0, color='red', linestyle='--')
    #         plt.axhline(-1.0, color='blue', linestyle='--')
    #         plt.axhline(-2.0, color='red', linestyle='--')
    #         plt.show()

    #     return sum(returns)

    def __threshold(self,signal1, signal2,stop_loss=4,entry=2,close=0):


        beta = OLS(signal2, signal1).fit().params[0]
        spread = signal2-beta*signal1

        normalized_spread = zscore(spread)
        standard_devitation = np.std(normalized_spread)

        close=0

        zs = normalized_spread
        returns=[]
        open_position=False
        initial_value=0
        p=0
        l=0
        loss=False
        for i in range(len(normalized_spread)):
            if open_position:
                if zs[i]>stop_loss and not below or  zs[i]<-stop_loss and below:
                    open_position=False
                    loss=True
                    l+=1
                    returns.append(-abs(spread[i]-initial_value))
                    
                elif zs[i]>close and below or zs[i] < close and not below:   
                    open_position=False             
                    p+=1
                    returns.append(abs(spread[i]-initial_value))

                else:
                    returns.append(0) 
            else:
                if (zs[i]>entry and zs[i]>0  or zs[i]<-entry and zs[i]<0) and not loss:
                    open_position=True
                    below=False if  zs[i]>0 else True
                    initial_value=spread[i]
                    
                   
                if (zs[i]<entry and zs[i]>0  or zs[i]>-entry and zs[i]<0) and loss:
                    open_position=True
                    below=False if  zs[i]>0 else True
                    loss=False
                    initial_value=spread[i]
                    
                   


        return sum(returns)

    def __moving_average(self,signal1, signal2):

        "https://www.quantrocket.com/codeload/quant-finance-lectures/quant_finance_lectures/Lecture42-Introduction-to-Pairs-Trading.ipynb.html"

        window = 10
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
        spread_mavg30 = spread.rolling(window).mean()
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
        p=0
        l=0
        for i in range(window,len(spread)):
            if open_position:
                if zscore_30_1[i]>3.0 or  zscore_30_1[i]<-3.0:
                    open_position=False
                    returns.append(-abs(spread[i]-entry))
                    l+=1
                elif zscore_30_1[i]>0 and direction or zscore_30_1[i] < 0 and not direction:   
                    open_position=False             
                    returns.append(abs(spread[i]-entry))
                    p+=1
                else:
                    returns.append(0)
            else:
                if zscore_30_1[i]>2.0 or  zscore_30_1[i]<-2.0:
                    open_position=True
                    entry=spread[i]
                    direction=False if  zscore_30_1[i]>2.0 else True
                    
                returns.append(0)
                
        print('profit positions=',p)
        print('stop loss positions=',l)
        if(self.__PLOT):

            plt.plot(spread_mavg1.index, spread_mavg1.values)
            plt.plot(spread_mavg30.index, spread_mavg30.values)
            plt.legend(['1 Day Spread MAVG', '30 Day Spread MAVG'])
            plt.ylabel('Spread')
            plt.show()

            plt.plot(zscore_30_1.index, zscore_30_1.values)
            plt.axhline(0, color='black')
            plt.axhline(1.0, color='blue', linestyle='--')
            plt.axhline(2.0, color='red', linestyle='--')
            plt.axhline(-1.0, color='blue', linestyle='--')
            plt.axhline(-2.0, color='red', linestyle='--')
            plt.show()

        return sum(returns)

    def run_simulation(self,model,verbose=False):

        function = {'MA':self.__moving_average,'TH':self.__threshold}
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

#antonio canelas sac ga tecnica de reconhecimento de padroes