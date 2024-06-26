

import numpy as np
import pandas as pd



from utils.utils import date_string, price_of_entire_component, compute_zscore, dataframe_interval, coint_spread, load_args,plot_positions,features_ts,plot_components,plot_forecasts


from utils.portfolio_optimization import ga_weights,uniform_portfolio,clean_weights


from utils.forecasting import calculate_metrics,lightgbm,arima

from statsmodels.tsa.stattools import coint,adfuller

from os.path import isfile, exists
from os import makedirs
import pickle

import math

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

PORTFOLIO_INIT = 1
NB_TRADING_DAYS = 252
CLOSE_INACTIVITY = 252

LONG_SPREAD = 1
SHORT_SPREAD = -1
CLOSE_POSITION = 0


class TradingPhase:

    """
    A class used to represent TradingPhase models

    """

    def __init__(self, data,sector):

        self.__pairs = []  # List of pairs
        self.__data = data  # Price series
        self.__tickers = data.keys()  # Data tickers
        self.__INIT_VALUE = PORTFOLIO_INIT
        self.__sector = sector
        self.counter=np.zeros((2,))
        try:
            self.__data.index=self.__data.index.strftime('%Y-%m-%d')
        except:
            pass

    def set_pairs(self, pairs):

        self.__pairs = pairs  # Set simulation pairs

    def set_dates(self, train_start, train_end, test_start, test_end):

        # Set dates for train/test
        self.__train_start = date_string(train_start)
        self.__train_end = date_string(train_end)
        self.__test_start = date_string(test_start)
        self.__test_end = date_string(test_end)

    def model_performance(self,model):


        if model == "FA":
            
            return pd.Series({
                    **calculate_metrics(self.forecast, self.y_true,self.y_fc_trend,self.y_true_trend)
                })
        else:
            return 0


    def __portfolio_data(self):

        args=load_args("TRADING")
        
        weights=args.get('weights')

        N=len(self.__pairs)

        self.__spread_train=pd.DataFrame()
        self.__spread_test=pd.DataFrame()
        self.__spread_full=pd.DataFrame()
        self.__c1_train = pd.DataFrame()
        self.__c2_train = pd.DataFrame()
        self.__c1_test = pd.DataFrame()
        self.__c2_test =pd.DataFrame()
        self.__c1_full = pd.DataFrame()
        self.__c2_full = pd.DataFrame()
        portfolio_returns= pd.DataFrame()
        self.__beta=np.empty(shape=(N,))
        self.__entry_close=np.empty(shape=(4,N))
        


        for id,(component1, component2) in enumerate(self.__pairs):  # get components for each pair

            # convert component1 and component2 to strings
            component1_str = '_'.join(component1)
            component2_str = '_'.join(component2)
            # Extract tickers in each component
            component1 = [(ticker in component1) for ticker in self.__tickers]
            component2 = [(ticker in component2) for ticker in self.__tickers]

            # Get one series for each component
            c1 = price_of_entire_component(self.__data, component1)
            c2 = price_of_entire_component(self.__data, component2)

            # Get series between train/test/full date intervals
            c1_train = dataframe_interval(
                self.__train_start,self.__train_end, c1)
            c2_train = dataframe_interval(
                self.__train_start,self.__train_end, c2)
            c1_test = dataframe_interval(
                self.__test_start, self.__test_end, c1)
            c2_test = dataframe_interval(
                self.__test_start, self.__test_end, c2)
            c1_full = dataframe_interval(
                self.__train_start, self.__test_end, c1)
            c2_full = dataframe_interval(
                self.__train_start, self.__test_end, c2)

            # Get beta coefficient and spread for train/test/full
            beta, spread_train,spread_test,spread_full = coint_spread(c1_train, c2_train,c1_test,c2_test,c1_full,c2_full)

            self.__spread_train[f'{component1_str}_{component2_str}'] = spread_train
            self.__spread_test[f'{component1_str}_{component2_str}'] = spread_test
            self.__spread_full[f'{component1_str}_{component2_str}'] = spread_full
            self.__c1_train[f'{component1_str}_{component2_str}'] = c1_train
            self.__c2_train[f'{component1_str}_{component2_str}'] = c2_train
            self.__c1_test[f'{component1_str}_{component2_str}'] = c1_test
            self.__c2_test[f'{component1_str}_{component2_str}'] = c2_test
            self.__c1_full[f'{component1_str}_{component2_str}'] = c1_full
            self.__c2_full[f'{component1_str}_{component2_str}'] = c2_full
            self.__beta[id] = beta

        # Get entry/exit thresholds, can change for new strategy
        vars_th= load_args("TH")
        self.__entry_close[0,:],self.__entry_close[1,:],self.__entry_close[2,:],self.__entry_close[3,:], self.__window= vars_th.get('entry_l'),vars_th.get('close_l'),vars_th.get('entry_s'),vars_th.get('close_s'),vars_th.get('window') 
 
        for id,(component1, component2) in enumerate(self.__pairs):  # get components for each pair
            # convert component1 and component2 to strings
            component1_str = '_'.join(component1)
            component2_str = '_'.join(component2)
            entry_l,close_l,entry_s,close_s=self.__entry_close[:,id]
            spread_train=self.__spread_train.iloc[:,[id]].squeeze()         
            c1_train=self.__c1_train.iloc[:,[id]].squeeze()         
            c2_train=self.__c2_train.iloc[:,[id]].squeeze()
            
            trade_array=self.__threshold(spread_full=spread_full,spread_test=spread_test,spread_train=spread_train,is_train=True,entry_l=entry_l,close_l=close_l,entry_s=entry_s,close_s=close_s,window=self.__window)
            returns=pd.Series(self.__trade_spread(c1_test=c1_train, c2_test=c2_train, trade_array=trade_array, FIXED_VALUE=self.__INIT_VALUE/N,beta=beta,**args)[2],index=spread_train.index)
            portfolio_returns[f'{component1_str}_{component2_str}'] = returns
            

        if N<=1:
            return [1]
        elif weights == "ga":
            return clean_weights(ga_weights(portfolio_returns,self.__spread_train,self.__sector,self.__window))
    
        elif weights == "u":
            return uniform_portfolio(N)
        
        else:
            return uniform_portfolio(N)
       
            
    def __force_close(self, decision_array):

        count = 0
        for day in range(1, len(decision_array)):  # Iterate trade_array

            if count >= CLOSE_INACTIVITY:  # Count reached non convergence threshold
                day_aux = day

                # Close all position until new decision is found in trade_array
                while decision_array[day_aux] == decision_array[day - 1]:
                    decision_array[day_aux] = CLOSE_POSITION
                    day_aux += 1  # next day

                    if day_aux == len(decision_array):  # end of trade array
                        break
            if decision_array[day] == decision_array[day-1]:  # same decision is found

                # Long/short decision, continue count
                if decision_array[day] == LONG_SPREAD or decision_array[day] == SHORT_SPREAD:
                    count += 1

            else:  # reset counter
                count = 0

        return decision_array
    

    
   
    def __trade_spread(self, c1_test, c2_test, trade_array, FIXED_VALUE=1000,transaction_cost=0.05,**kwargs):
        
        # Close all positions in the last day of the trading period whether they have converged or not
        trade_array.iloc[-1] = CLOSE_POSITION

        # define trading costs
        fixed_costs_per_trade = (transaction_cost) / 100  # remove percentage


        # 2 positions, one for each component of the pair
        # The first position concerns the fist component, c1, and 2nd the c2
        stocks_in_hand = np.zeros(2)
        # tracks the evolution of the balance day by day
        cash_in_hand = np.zeros(len(trade_array))
        cash_in_hand[0] = FIXED_VALUE  # starting balance
        portfolio_value = np.zeros(len(trade_array))
        portfolio_value[0] = cash_in_hand[0]

        n_trades = 0  # how many trades were made?
        # how many profitable/unprofitable trades were made
        profitable_unprofitable = np.zeros(2)

        # how many days has this position been open?
        days_open = np.zeros(len(trade_array))
        profit_loss=[]
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
      


            # extreme case where portfolio value of pair is smaller than 10% of init value - termiate trading
            if portfolio_value[day]<portfolio_value[0]*0.1:
                trade_array[day:]=CLOSE_POSITION
                
            # the state has changed and the TS is called to act
            if trade_array[day] != trade_array[day - 1]:

                n_trades += 1

                sale_value = stocks_in_hand[0] * c1_test[day]+stocks_in_hand[1] * c2_test[day]
                # apply transaction costs
                cash_in_hand[day] += sale_value * (1 - fixed_costs_per_trade)

                # both positions were closed
                stocks_in_hand[0] = stocks_in_hand[1] = 0

                days_open[day] = 0  # the new position was just opened

                if sale_value > 0:
                    profitable_unprofitable[0] += 1  # profit
                    profit_loss.append(1)
                elif sale_value < 0:
                    profitable_unprofitable[1] += 1  # loss
                    profit_loss.append(0)

                if decision == SHORT_SPREAD:  # if the new decision is to SHORT the spread
                    # if the previous trades lost money I have less than FIXED VALUE to invest
                    value_to_buy = min(FIXED_VALUE,cash_in_hand[day]) 
                    # long c2
                    cash_in_hand[day] += -value_to_buy 
                    stocks_in_hand[1] = value_to_buy  / c2_test[day] 
                    # short c1
                    cash_in_hand[day] += value_to_buy
                    stocks_in_hand[0] = -value_to_buy  / c1_test[day]
                    # apply transaction costs (with 2 operations made: short + long)
                    cash_in_hand[day] -= value_to_buy*fixed_costs_per_trade*2

                elif decision == LONG_SPREAD:  # if the new decision is to LONG the spread
                    value_to_buy = min(FIXED_VALUE, cash_in_hand[day]) 
                    # long c1
                    cash_in_hand[day] += -value_to_buy 
                    stocks_in_hand[0] = value_to_buy  / c1_test[day] 
                    # short c2
                    cash_in_hand[day] += value_to_buy 
                    stocks_in_hand[1] = -value_to_buy / c2_test[day]  
                    # apply transaction costs (with 2 operations made: short + long)
                    cash_in_hand[day] -= value_to_buy*fixed_costs_per_trade*2

            # short rental costs are applied daily!
            # means there's an open position
            if stocks_in_hand[0] != 0 or stocks_in_hand[1] != 0:
                days_open[day] += 1
            # at the end of the day, the portfolio value takes in consideration the value of the stocks in hand
            portfolio_value[day] = cash_in_hand[day] + \
                stocks_in_hand[0] * c1_test[day] + stocks_in_hand[1] * c2_test[day]
  

        return n_trades, cash_in_hand, portfolio_value, days_open, profitable_unprofitable , profit_loss
    
    
    def __threshold(self, spread_full, spread_test, spread_train,is_train,c1_test=0,c2_test=0,entry_l=-2, close_l=0, entry_s=2, close_s=0,stop=2,window=21,plot=False, **kwargs):

        if is_train:
            spread=np.array((spread_train-spread_train.mean())/spread_train.std())
            spread_mean = spread_train.rolling(window=window).mean().to_numpy()
            spread_std = spread_train.rolling(window=window).std().to_numpy()
            spread=np.array((spread_train-spread_mean)/spread_std)
            spread[:window-1]=0
        else:
            # Norm spread
            spread, _, _, _ = compute_zscore(spread_full, spread_test,window)
            # spread=c1_test-c2_test
            # spread=np.array((spread - spread.mean()) / spread.std())

        # Get entry/exit points
        longs_entry = spread < entry_l
        longs_exit = spread > close_l
        shorts_entry = spread > entry_s
        shorts_exit = spread < close_s

        # In the first day of trading no trades will be made to stabilize the spread
        stabilizing_threshold = 1
        longs_entry[:stabilizing_threshold] = False
        longs_exit[:stabilizing_threshold] = False
        shorts_entry[:stabilizing_threshold] = False
        shorts_exit[:stabilizing_threshold] = False

        # numerical_units long/short - equivalent to the long/short_entry arrays but with integers instead of booleans
        num_units_long = pd.Series([np.nan for i in range(len(spread))])
        num_units_short = pd.Series([np.nan for i in range(len(spread))])

        num_units_long[longs_entry] = LONG_SPREAD
        num_units_long[longs_exit] = CLOSE_POSITION
        num_units_short[shorts_entry] = SHORT_SPREAD
        num_units_short[shorts_exit] = CLOSE_POSITION


        # a bit redundant, the stabilizing threshold ensures this
        num_units_long[0] = CLOSE_POSITION
        num_units_short[0] = CLOSE_POSITION

        # completes the array by propagating the last valid observation
        num_units_long = num_units_long.fillna(method='ffill')
        num_units_short = num_units_short.fillna(method='ffill')

        # concatenation of both arrays in a single decision array
        num_units = num_units_long + num_units_short
        trade_array = pd.Series(data=num_units.values)
        trade_array.iloc[-1] = CLOSE_POSITION

        return trade_array
    

    def __forecasting_algorithm(self, spread_train, spread_full, spread_test, verbose=True, plot=True,
                                horizon=1, batch=1,model=['arima'],window=21,**kwargs):
        verbose=False
        plot=False

        norm,_,_,_=compute_zscore(spread_full,spread_test,window)
        
        X = features_ts(spread_full).dropna()
        Y = pd.DataFrame([X['y'].shift(-i).values for i in range(1, horizon + 1)]).T
        offset = X.index.get_loc(spread_test.index[0])
        y_price=spread_test.values
        y_true= pd.DataFrame(Y[offset:].values,index=spread_test.index)
        y_true.columns = [spread_test.name]*(len(y_true.columns))

        compute_fc=False
        file='results/'+'forecasts.pkl'
        date=[]
        fc_list=[]

        for key in model:
            date.append(spread_test.index[0]+'_'+spread_test.index[-1]+'_'+key+'_'+spread_test.name)
        isExist = exists(file)

        if isExist:
            # Load the data from a pickle file
            with open(file, 'rb') as f:
                forecasts_algo = pickle.load(f)
        else:
            forecasts_algo={}


        for key in range(len(date)):
            if compute_fc or (date[key] not in forecasts_algo or forecasts_algo[date[key]] is None or forecasts_algo[date[key]].shape[1] < horizon):

                # Initialize fcts with the existing forecasts for the key, if any
                fcts = forecasts_algo[date[key]] if date[key] in forecasts_algo and not compute_fc else np.empty((len(spread_test), 0))


                if model[key] == 'lightgbm':
                    # Compute only for the new columns
                    for h in range(fcts.shape[1], Y.shape[1]):
                        new_fct = lightgbm(X, Y.iloc[:, h], offset, h+1, batch)
                        # Append the new forecast to fcts
                        fcts = np.hstack((fcts, new_fct.reshape(-1, 1)))
                        forecasts_algo.update({date[key]: fcts})
            
                if model[key] == 'arima':
                    fcts = arima(X, Y, offset, horizon)
                    forecasts_algo.update({date[key]: fcts})

                # Save the array to a pickle file
                with open(file, 'wb') as f:
                    pickle.dump(forecasts_algo, f)
            else:
                fcts = forecasts_algo[date[key]]

            fc_list.append(fcts)


        # Stack arrays along a new axis (axis=0) to create a 3D array
        sliced_forecasts = np.array([forecast[:, :horizon] for forecast in fc_list])

        # Compute the mean along the first axis (axis=0)
        fc = np.mean(sliced_forecasts, axis=0)

        y_true_trend = np.where(np.mean(y_true.values,axis=1)>y_price, 1, -1)
        y_fc_trend = np.where(np.mean(fc,axis=1)>y_price, 1, -1)

        try:     
            self.forecast=np.append(self.forecast,fc[:-horizon],axis=0)
            self.y_true=np.append(self.y_true,y_true[:-horizon],axis=0)
            self.y_true_trend=np.append(self.y_true_trend,y_true_trend[:-horizon],axis=0)
            self.y_fc_trend=np.append(self.y_fc_trend,y_fc_trend[:-horizon],axis=0)
        except:
            self.forecast,self.y_true=fc[:-horizon], y_true[:-horizon],
            self.y_fc_trend,self.y_true_trend=y_fc_trend[:-horizon],y_true_trend[:-horizon]

        error=pd.Series({
                    **calculate_metrics(fc[:-horizon], y_true[:-horizon],y_fc_trend[:-horizon],y_true_trend[:-horizon])
                })

        if verbose:
            print(error)

        decision_array = pd.Series([np.nan for i in range(len(spread_test))])


        args=load_args("TH")
        entry_l = args.get('entry_l')  
        entry_s = args.get('entry_s')  
        close_l = args.get('close_l')  
        close_s = args.get('close_s')  

        position = CLOSE_POSITION
        percentage_l=0.5
        percentage_s=0.5
        alpha_l=entry_l-percentage_s*entry_l
        alpha_s=entry_s-(percentage_l*(entry_s))

        for today in range(1,len(spread_test)-1):
        
            delta=(np.mean(fc[today])-y_price[today])/np.abs(y_price[today])

            if (position == LONG_SPREAD and norm[today]>close_l)or (position == SHORT_SPREAD and norm[today]<close_s):
                decision_array[today]=position=CLOSE_POSITION
            if norm[today]>alpha_s:
                if delta<0:
                    decision_array[today]=position=SHORT_SPREAD

            elif norm[today]<alpha_l:
                if delta>0:
                    decision_array[today]=position=LONG_SPREAD


        decision_array.iloc[-1] = decision_array.iloc[0] =CLOSE_POSITION
        decision_array=decision_array.fillna(method='ffill')

        if plot:
            plot_forecasts(y_true,sliced_forecasts,horizon,model)


        return decision_array
   

    def run_simulation(self, model):

        verbose=False
        plot=False

        # Select function
        function = {'TH': self.__threshold,'FA': self.__forecasting_algorithm}

        # Create function dictionary
        args = load_args(model)

        # Trading variables dictionary
        # Update 
        args.update(load_args("TRADING"))

        # Initialize stats variables
        n_pairs = len(self.__pairs)

        if n_pairs==0:
            aux_pt_value=aux_cash=np.full(len(dataframe_interval(self.__test_start, self.__test_end, self.__data)),self.__INIT_VALUE)

        n_non_convergent_pairs = 0
        profit = 0
        loss = 0
        profit_loss_trade = np.zeros(2)
        total_trades = 0

        weights=self.__portfolio_data()

        window=self.__window
        is_train=False

        

        if verbose:
            print(weights)
            print(self.__test_start)

        for pair_id in range(n_pairs):  # get components for each pair

            if weights[pair_id]==0:
                continue

            entry_l,close_l,entry_s,close_s=self.__entry_close[:,pair_id] 
            spread_train=self.__spread_train.iloc[:,[pair_id]].squeeze()
            spread_test=self.__spread_test.iloc[:,[pair_id]].squeeze()
            spread_full=self.__spread_full.iloc[:,[pair_id]].squeeze()            
            c1_train=self.__c1_train.iloc[:,[pair_id]].squeeze()         
            c2_train=self.__c2_train.iloc[:,[pair_id]].squeeze()         
            c1_test=self.__c1_test.iloc[:,[pair_id]].squeeze()        
            c2_test=self.__c2_test.iloc[:,[pair_id]].squeeze()        
            c1_full=self.__c1_full.iloc[:,[pair_id]].squeeze()        
            c2_full=self.__c2_full.iloc[:,[pair_id]].squeeze()
            FIXED_VALUE = self.__INIT_VALUE*weights[pair_id]
            beta=self.__beta[pair_id]

            # Create a dictionary of the current pair's variables
            vars_pair = {k: v for k, v in locals().items() if k in 
                        ['c1_train', 'c2_train', 'c1_test', 'c2_test', 'c1_full', 'c2_full',
                        'spread_train', 'spread_full', 'spread_test', 
                        'entry_l', 'close_l', 'entry_s', 'close_s',
                        'FIXED_VALUE', 'window','beta', 'pair_id','is_train']}
            

            # Update the arguments
            args.update(vars_pair)

            # Apply trading model and get trading decision array
            trade_array = function[model](**args)

            # Force close non convergent positions
            # trade_array = self.__force_close(trade_array)


            # # Apply trading rules to trade decision array
            n_trades, cash, portfolio_value, days_open, profitable_unprofitable, profit_loss = self.__trade_spread(trade_array=trade_array,**args)

            

            # Evaluate pair performance
            pair_performance = portfolio_value[-1]/portfolio_value[0] * 100

            if verbose:
                print(spread_train.name,':',pair_performance)

            if plot:
                plot_positions(spread_test,spread_full,trade_array,window,profit_loss,portfolio_value,c1_test,c2_test)
                plot_components(c1_train,c2_train)



            if pair_performance > 100:
                profit += 1
            else:
                loss += 1

            total_trades += n_trades

            try:  # Add portfolio variables
                aux_pt_value += portfolio_value
                aux_cash += cash

            except:  # First pair, init portfolio variables
                aux_pt_value = portfolio_value
                aux_cash = cash

            # non convergent pair
            if days_open[-2] > 0:
                n_non_convergent_pairs += 1

            profit_loss_trade += profitable_unprofitable

            

        # TradingPhase dictionary
        stats = {
            "model": model,
            "portfolio_start": self.__INIT_VALUE,
            "portfolio_end": aux_pt_value[-1],
            "portfolio_value": list(aux_pt_value),
            "trading_start": self.__test_start,
            "trading_end": self.__test_end,
            "weights": weights,
            "cash": list(aux_cash),
            "n_pairs": (profit+loss),
            "profit_pairs": profit,
            "loss_pairs": loss,
            "profit_trades": int(profit_loss_trade[0]),
            "loss_trades": int(profit_loss_trade[1]),
            "non_convergent_pairs": n_non_convergent_pairs,
        }

        # Change portfolio init value for next simulation
        self.__INIT_VALUE = aux_pt_value[-1]

        return stats