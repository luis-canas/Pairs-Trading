

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

    def __threshold(self,signal1, signal2,stop_loss=4,entry=2,close=0):


        beta = OLS(signal2, signal1).fit().params[0]
        spread = signal2-beta*signal1


        # Compute the z score for each day
        zs = zscore(spread)

        returns=[]
        open_position=False
        initial_value=0
        p=0
        l=0
        for i in range(len(spread)):
            if open_position:
                if zs[i]>stop_loss or  zs[i]<-stop_loss:
                    open_position=False
                    returns.append(-abs(spread[i]-initial_value))
                    l+=1
                elif zs[i]>close and rising or zs[i] < close and not rising:   
                    open_position=False             
                    returns.append(abs(spread[i]-initial_value))
                    p+=1
                else:
                    returns.append(0)
            else:
                if zs[i]>entry or  zs[i]<-entry:
                    open_position=True
                    initial_value=spread[i]
                    rising=False if  zs[i]>2.0 else True
                    
                returns.append(0)

        print('profit positions=',p)
        print('stop loss positions=',l)
        if(self.__PLOT):

            plt.plot(zs.index, zs.values)
            plt.plot(zs.index, zs.values)
            plt.legend(['1 Day Spread MAVG', '30 Day Spread MAVG'])
            plt.ylabel('Spread')
            plt.show()

            plt.plot(zs.index, zs.values)
            plt.axhline(0, color='black')
            plt.axhline(1.0, color='blue', linestyle='--')
            plt.axhline(2.0, color='red', linestyle='--')
            plt.axhline(-1.0, color='blue', linestyle='--')
            plt.axhline(-2.0, color='red', linestyle='--')
            plt.show()

        return sum(returns)

    def __threshold_model(self,signal1, signal2,stop_loss=4,entry=2,close=0):


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

        function = {'MA':self.__moving_average,'TH':self.__threshold_model}
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

    # def threshold_trading_HALF(b, verbose, normalization_period, close_if_inactive, entry, exit):

    #     total_portfolio_value = []
    #     total_cash = []

    #     for half_year_dict in b: 
    #         bem_sucedidos = 0
    #         mal_sucedidos = 0

    #         n_pairs = len(half_year_dict['PAIRS']) 
    #         testing_period = half_year_dict['Testing Period']
    #         year = int(testing_period[:4])

    #         if  "2017-01-01" in testing_period: #first testing year
    #             FIXED_VALUE = 1000 / n_pairs
    #         else:
    #             FIXED_VALUE = total_portfolio_value[-1] / n_pairs


    #         if "01-01" in testing_period:
    #             half = 'first'
    #         else:
    #             half = 'second'
            
    #         train_series, test_series, t_period, offset = split_data3(year-1, half, series)
            
    #         trader = class_TradingStage_v3.TradingStage()

    #         aux_pt_value = np.zeros(int(test_series.size / len(tickers))) 
    #         aux_cash = np.zeros(int(test_series.size / len(tickers))) 

    #         for pair in half_year_dict['PAIRS']:
    #             component_1 = pair[0]
    #             component_2 = pair[1]
                
    #         if verbose: print(year, 'pair:', component_1, ',', component_2)
    #             component_1 = [(ticker in component_1) for ticker in tickers]
    #             component_2 = [(ticker in component_2) for ticker in tickers]

    #         c1 = price_of_entire_component(test_series, component_1)
    #         c2 = price_of_entire_component(test_series, component_2)

    #         train_c1 = price_of_entire_component(train_series, component_1)
    #         train_c2 = price_of_entire_component(train_series, component_2)

    #         series_without_date = series.drop('Date', axis = 1)
    #         c1_full = price_of_entire_component(series_without_date,  component_1)
    #         c2_full = price_of_entire_component(series_without_date,  component_2)

    #         training_beta, _ = calculate_beta(train_c1, train_c2)

    #         test_spread = c1 - training_beta * c2
    #         train_spread = train_c1 - training_beta * train_c2
    #         full_spread = c1_full - training_beta*c2_full

    #         #normalized spread   #TEM QUE TER O BETA
    #         norm_spread, mean, std, t_spread = zscore_COMODEVESER(full_spread, test_spread, offset, normalization_period)

    #         decision_array = np.array(trader.threshold_decision_system( pd.Series(norm_spread), entry_level=entry, exit_level=exit))   

    #         c1_array = np.array(c1)
    #         c2_array = np.array(c2)

    #         force_close(decision_array, close_if_inactive)

    #         n_trades, cash, portfolio_value, days_open, decision_array = trader.trading_system(c1_array, c2_array, decision_array, FIXED_VALUE)

    #         pair_performance = portfolio_value[-1]/portfolio_value[0] * 100

    #         # print('last value', portfolio_value[-1])
    #         # print('first_value', portfolio_value[0])

    #         if verbose: print('Pair performance', pair_performance )
    #         if pair_performance > 100:
    #             bem_sucedidos += 1
    #         else:
    #             mal_sucedidos += 1

    #         aux_pt_value += portfolio_value
    #         aux_cash += cash

    #         yearly_performance = (aux_pt_value[-1]/(FIXED_VALUE * n_pairs)) * 100        

    #         total_portfolio_value += list(aux_pt_value)
    #         total_cash += list(aux_cash)

            
            
            
    #     total = np.array(total_portfolio_value)
    #     total_cash = np.array(total_cash)


    #     print("\n\n")
    #     plt.plot(total)
    #     plt.xlabel("time")
    #     plt.ylabel("portfolio_value")
    #     plt.title("evolution of the portfolio_value")
    #     plt.show()

    #     print('TOTAL PERFORMANCE = ', total[-1]/total[0])
#antonio canelas sac ga tecnica de reconhecimento de padroes