

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.api import OLS
from statsmodels.tsa.arima_model import ARMA, ARIMA
from tqdm import tqdm

from utils import *

PORTFOLIO_INIT=1000
LONG_SPREAD = 1
SHORT_SPREAD = -1
CLOSE_POSITION = 0

NB_TRADING_DAYS = 252

class TradingPhase:

    def __init__(self,data):

        self.__all_pairs=[]
        self.__data=data
        self.__tickers=data.keys()
        self.__portfolio_init=PORTFOLIO_INIT

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
        trade_array = pd.Series(data=num_units.values)

        return trade_array


    def __trade_spread(self, c1, c2, trade_array, FIXED_VALUE = 1000, commission = 0.08,  market_impact=0.2, short_loan=1):
        
        trade_array.iloc[-1] = CLOSE_POSITION  # Close all positions in the last day of the trading period whether they have converged or not

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

        n_pairs=len(all_pairs)
        self.__n_non_convergent_pairs = 0
        self.__profit = 0
        self.__loss = 0
        total_trades = 0

        FIXED_VALUE = self.__portfolio_init / n_pairs

        self.__total_portfolio_value = []
        self.__total_cash = [] 


        for i in tqdm(range(n_pairs)):

            pair=all_pairs[i]
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

            c1_full=dataframe_interval(self.__train_start,self.__test_end,c1)
            c2_full=dataframe_interval(self.__train_start,self.__test_end,c2)

            beta,_=coint_spread(c1_train,c2_train)

            full_spread=c1_full-beta*c2_full
            spread=c1_test-beta*c2_test

            
            norm_spread,_,_,_=compute_zscore(full_spread,spread)

            trade_array=self.__threshold(spread=norm_spread)
            
            n_trades, cash, portfolio_value, days_open, profitable_unprofitable=self.__trade_spread(c1=c1_test, c2=c2_test, trade_array=trade_array)

            pair_performance = portfolio_value[-1]/portfolio_value[0] * 100

            if verbose: print('Pair performance', pair_performance - 100, '%' )
            if pair_performance > 100:
                self.__profit += 1
            else:
                self.__loss += 1

            try:
                aux_pt_value += portfolio_value
                aux_cash += cash
                total_trades += n_trades
            except:
                aux_pt_value = portfolio_value
                aux_cash = cash
                total_trades = n_trades


            # non convergent pairs
            if days_open[-2] > 0:
                self.__n_non_convergent_pairs += 1


      
        self.__total_portfolio_value += list(aux_pt_value)
        self.__total_cash += list(aux_cash)

        self.__roi = (aux_pt_value[-1]/(FIXED_VALUE * n_pairs)) * 100


    def run_simulation(self,model,verbose=False,plot=False):

        function = {'TH':self.__threshold_model}

        function[model](verbose=verbose,plot=plot)

        info={"portfolio_init": self.__portfolio_init,
                "portfolio_value": self.__total_portfolio_value,
                "simulation_start": self.__test_start,
                "simulation_end": self.__test_end,
                "cash": self.__total_cash,
                "profit_pairs":self.__profit,
                "loss_pairs":self.__loss,
                "non_convergent_pairs":self.__n_non_convergent_pairs,
                "roi": self.__roi,
        }
        
        return info
      
#antonio canelas sac ga tecnica de reconhecimento de padroes