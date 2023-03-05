

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.api import OLS
from statsmodels.tsa.arima_model import ARMA, ARIMA

from utils import *

LONG_SPREAD = 1
SHORT_SPREAD = -1
CLOSE_POSITION = 0

NB_TRADING_DAYS = 252

class Trader:

    __PLOT=False

    def __init__(self,data):

        self.__all_pairs=[]
        self.__training_start=[]
        self.__training_end=[]
        self.__testing_start=[]
        self.__testing_end=[]
        self.__data=data
        self.__tickers=data.keys()

    def set_pairs(self,pairs):
        
        self.__all_pairs=pairs

    def set_dates(self,train_start,train_end,test_start,test_end):
        
        self.__train_start=date_string(train_start)
        self.__train_end=date_string(train_end)
        self.__test_start=date_string(test_start)
        self.__test_end=date_string(test_end)

    def __threshold(self,spread,stop_loss=4,entry=2,close=0):

        longs_entry = spread < -entry
        longs_exit = spread > -close

        shorts_entry = spread > entry
        shorts_exit = spread < close

        stabilizing_threshold = 5
        # in the first 5 days of trading no trades will be made to stabilize the spread
        longs_entry[:stabilizing_threshold] = False
        longs_exit[:stabilizing_threshold] = False
        shorts_entry[:stabilizing_threshold] = False
        shorts_exit[:stabilizing_threshold] = False

        #numerical_units long/short - equivalent to the long/short_entry arrays but with integers instead of booleans
        num_units_long = pd.Series([np.nan for i in range(len(spread))])
        num_units_short = pd.Series([np.nan for i in range(len(spread))])

        num_units_long[longs_entry] = LONG_SPREAD
        num_units_long[longs_exit] = CLOSE_POSITION
        num_units_short[shorts_entry] = SHORT_SPREAD
        num_units_short[shorts_exit] = CLOSE_POSITION

        #a bit redundant, the stabilizing threshold ensures this
        num_units_long[0] = CLOSE_POSITION
        num_units_short[0] = CLOSE_POSITION

        #completes the array by propagating the last valid observation
        num_units_long = num_units_long.fillna(method='ffill')
        num_units_short = num_units_short.fillna(method='ffill')

        #concatenation of both arrays in a single decision array
        num_units = num_units_long + num_units_short
        trade_array = pd.Series(data=num_units.values, index=spread.index)

        return trade_array


    def __trade_spread(self, c1, c2, trade_array, FIXED_VALUE = 1000, commission = 0.08,  market_impact=0.2, short_loan=1):
        
        trade_array[-1] = CLOSE_POSITION  # Close all positions in the last day of the trading period whether they have converged or not

        # define trading costs
        fixed_costs_per_trade = (commission + market_impact) / 100  # remove percentage
        short_costs_per_day = FIXED_VALUE * (short_loan / NB_TRADING_DAYS) / 100  # remove percentage

        # 2 positions, one for each component of the pair
        stocks_in_hand = np.zeros(2)  # The first position concerns the fist component, c1, and 2nd the c2
        cash_in_hand = np.zeros(len(trade_array))  # tracks the evolution of the balance day by day
        cash_in_hand[0] = FIXED_VALUE  # starting balance
        portfolio_value = np.zeros(len(trade_array))
        portfolio_value[0] = cash_in_hand[0]

        n_trades = 0  # how many trades were made?
        profitable_unprofitable = np.zeros(2)  # how many profitable/unprofitable trades were made

        days_open = np.zeros(len(trade_array))  # how many days has this position been open?

        for day, decision in enumerate(trade_array):
            # the first day of trading is excluded to stabilize the spread
            # and avoid  accessing positions out of range in the decision array when executing decision_array[day-1]
            if day == 0:
                continue  # skip the first day as mentioned above

            # at the beginning of the day we still have the cash we had the day before
            cash_in_hand[day] = cash_in_hand[day - 1]
            portfolio_value[day] = portfolio_value[day - 1]

            # at the beginning of the day the position hasn't been altered
            days_open[day] = days_open[day-1]

            if trade_array[day] != trade_array[day - 1]:  # the state has changed and the TS is called to act

                n_trades += 1


                sale_value = stocks_in_hand[0] * c1[day] + stocks_in_hand[1] * c2[day]
                cash_in_hand[day] += sale_value * (1 - 2*fixed_costs_per_trade) # 2 closes, so 2*transaction costs

                stocks_in_hand[0] = stocks_in_hand[1] = 0  # both positions were closed

                days_open[day] = 0  # the new position was just opened

                if sale_value > 0:
                    profitable_unprofitable[0] += 1  # profit
                elif sale_value < 0:
                    profitable_unprofitable[1] += 1  # loss


                if decision == SHORT_SPREAD:  # if the new decision is to SHORT the spread
                    value_to_buy = min(FIXED_VALUE, cash_in_hand[day]) # if the previous trades lost money I have less than FIXED VALUE to invest
                    # long c2
                    cash_in_hand[day] += -value_to_buy
                    stocks_in_hand[1] = value_to_buy / c2[day]
                    # short c1
                    cash_in_hand[day] += value_to_buy
                    stocks_in_hand[0] = -value_to_buy / c1[day]
                    # apply transaction costs (with 2 operations made: short + long)
                    cash_in_hand[day] -= 2*value_to_buy*fixed_costs_per_trade

                elif decision == LONG_SPREAD:  # if the new decision is to LONG the spread
                    value_to_buy = min(FIXED_VALUE, cash_in_hand[day])
                    # long c1
                    cash_in_hand[day] += -value_to_buy
                    stocks_in_hand[0] = value_to_buy / c1[day]
                    # short c2
                    cash_in_hand[day] += value_to_buy
                    stocks_in_hand[1] = -value_to_buy / c2[day]
                    # apply transaction costs (with 2 operations made: short + long)
                    cash_in_hand[day] -= 2 * value_to_buy * fixed_costs_per_trade

            # short rental costs are applied daily!
            if stocks_in_hand[0] != 0 or stocks_in_hand[1] != 0:  # means there's an open position
                cash_in_hand[day] -= short_costs_per_day
                days_open[day] += 1
            # at the end of the day, the portfolio value takes in consideration the value of the stocks in hand
            portfolio_value[day] = cash_in_hand[day] + stocks_in_hand[0] * c1[day] + stocks_in_hand[1] * c2[day]

        return n_trades, cash_in_hand, portfolio_value, days_open, profitable_unprofitable
    
    def __threshold_model(self,verbose,plot):


        data=self.__data
        tickers=self.__tickers
        all_pairs=self.__all_pairs

        for pair in all_pairs:

            component1=pair[0]
            component2=pair[1]

            component1 = [(ticker in component1) for ticker in tickers]
            component2 = [(ticker in component2) for ticker in tickers]

            c1 = price_of_entire_component(data, component1)
            c2 = price_of_entire_component(data, component2)

            c1_train=dataframe_interval(self.__train_start,self.__train_end,c1)
            c2_train=dataframe_interval(self.__train_start,self.__train_end,c2)

            c1_test=dataframe_interval(self.__test_start,self.__test_end,c1)
            c2_test=dataframe_interval(self.__test_start,self.__test_end,c2)

            beta,_=coint_spread(c1_train,c2_train)

            spread=c1-beta*c2

            norm_spread=compute_zscore(spread)

            trade_array=self.__threshold(spread=norm_spread)
            
            n_trades, cash_in_hand, portfolio_value, days_open, profitable_unprofitable=self.__trade_spread(spread)

        return 1

   

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

    def run_simulation(self,model,verbose=False,plot=False):

        function = {'MA':self.__moving_average,'TH':self.__threshold_model}
        summary={'Returns':0}

        function[model](verbose=verbose,plot=plot)
        # if verbose:
        #     print("\n************************************************\n",
        #             "\nModel: ",model)

        # for signal1,signal2 in self.__all_pairs:
            

        #     returns=function[model](self.__data[signal1],self.__data[signal2])
        #     summary['Returns']+=returns

        #     if verbose:
        #         print("Pair ({}-{}) returns: {}".format(signal1,signal2,returns))

                


        
        # print("Portfolio returns: ",summary['Returns'],
        #         "\n\n************************************************\n")

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