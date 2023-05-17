

import numpy as np
import pandas as pd


from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.ref_dirs import get_reference_directions

from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

from pymoo.optimize import minimize

from utils.utils import date_string, price_of_entire_component, compute_zscore, dataframe_interval, coint_spread, load_args, plot_positions
from utils.symbolic_aggregate_approximation import pattern_distance,get_best_distance,get_best_patterns

from utils.genetic_algorithm import SaxObjectives,SaxObjectivesGA

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

PORTFOLIO_INIT = 1000
NB_TRADING_DAYS = 252
CLOSE_INACTIVITY = 252

LONG_SPREAD = 1
SHORT_SPREAD = -1
CLOSE_POSITION = 0


class TradingPhase:

    """
    A class used to represent TradingPhase models

    """

    def __init__(self, data):

        self.__pairs = []  # List of pairs
        self.__data = data  # Price series
        self.__tickers = data.keys()  # Data tickers
        self.__INIT_VALUE = PORTFOLIO_INIT

    def set_pairs(self, pairs):

        self.__pairs = pairs  # Set simulation pairs

    def set_dates(self, train_start, train_end, test_start, test_end):

        # Set dates for train/test
        self.__train_start = date_string(train_start)
        self.__train_end = date_string(train_end)
        self.__test_start = date_string(test_start)
        self.__test_end = date_string(test_end)

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

    def __trade_spread(self, c1, c2, trade_array, FIXED_VALUE=1000, commission=0.08,  market_impact=0.2, short_loan=1, **kwargs):

        # Close all positions in the last day of the trading period whether they have converged or not
        trade_array.iloc[-1] = CLOSE_POSITION

        # define trading costs
        fixed_costs_per_trade = (
            commission + market_impact) / 100  # remove percentage
        short_costs_per_day = FIXED_VALUE * \
            (short_loan / NB_TRADING_DAYS) / 100  # remove percentage

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

            # the state has changed and the TS is called to act
            if trade_array[day] != trade_array[day - 1]:

                n_trades += 1

                sale_value = stocks_in_hand[0] * \
                    c1[day] + stocks_in_hand[1] * c2[day]
                # 2 closes, so 2*transaction costs
                cash_in_hand[day] += sale_value * (1 - 2*fixed_costs_per_trade)

                # both positions were closed
                stocks_in_hand[0] = stocks_in_hand[1] = 0

                days_open[day] = 0  # the new position was just opened

                if sale_value > 0:
                    profitable_unprofitable[0] += 1  # profit
                elif sale_value < 0:
                    profitable_unprofitable[1] += 1  # loss

                if decision == SHORT_SPREAD:  # if the new decision is to SHORT the spread
                    # if the previous trades lost money I have less than FIXED VALUE to invest
                    value_to_buy = min(FIXED_VALUE, cash_in_hand[day])
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
                    cash_in_hand[day] -= 2 * \
                        value_to_buy * fixed_costs_per_trade

            # short rental costs are applied daily!
            # means there's an open position
            if stocks_in_hand[0] != 0 or stocks_in_hand[1] != 0:
                cash_in_hand[day] -= short_costs_per_day
                days_open[day] += 1
            # at the end of the day, the portfolio value takes in consideration the value of the stocks in hand
            portfolio_value[day] = cash_in_hand[day] + \
                stocks_in_hand[0] * c1[day] + stocks_in_hand[1] * c2[day]

        return n_trades, cash_in_hand, portfolio_value, days_open, profitable_unprofitable

    def __threshold(self, spread_full, spread_test, entry=2, close=0, plot=True, **kwargs):

        # Norm spread
        spread, _, _, _ = compute_zscore(spread_full, spread_test)

        # Get entry/exit points
        longs_entry = spread < -entry
        longs_exit = spread > -close
        shorts_entry = spread > entry
        shorts_exit = spread < close

        # In the first 5 days of trading no trades will be made to stabilize the spread
        stabilizing_threshold = 5
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

        if plot:
            plot_positions(spread, trade_array)

        return trade_array

    def __forecasting_algorithm(self, spread_train, spread_full, spread_test, c1_train, c2_train, c1_test, c2_test, FIXED_VALUE=1000, commission=0.08,  market_impact=0.2, short_loan=1,
                                gen=100, pop=50, w_size=20, alphabet_size=10, verbose=True, **kwargs):

        # Create lagged variables
        x_train = spread_train
        y_train = spread_train.shift(-1)
        x_train, y_train = x_train[:-1], y_train[:-1]
        x_test = spread_test
        y_test = spread_test.shift(-1)
        x_test, y_test = x_test[:-1], y_test[:-1]

        eval_set = [(x_train, y_train), (x_test, y_test)]

        parameters = {
            'n_estimators': [400],
            'learning_rate': [0.05],
            'max_depth': [8],
            'gamma': [0.005],
            'random_state': [42]
        }

        model = xgb.XGBRegressor(objective='reg:squarederror',  verbose=False)
        clf = GridSearchCV(model, parameters)
        clf.fit(x_train, y_train, eval_set=eval_set,
                verbose=False, eval_metric=["rmse"])

        model = xgb.XGBRegressor(
            **clf.best_params_, objective='reg:squarederror')
        model.fit(x_train, y_train, eval_set=eval_set, verbose=False,
                  early_stopping_rounds=5, eval_metric=["rmse"])

        spread = spread_test
        pred_spread = model.predict(x_test)

        decision_array = pd.Series([np.nan for i in range(len(spread))])

        # define trading costs
        fixed_costs_per_trade = (
            commission + market_impact) / 100  # remove percentage
        short_costs_per_day = FIXED_VALUE * \
            (short_loan / NB_TRADING_DAYS) / 100  # remove percentage

        # each play requires 2*(long+short) operations = 4 chargeable operations
        # as each play only lasts 1 day, the short loan costs = 1 * short_costs_per_day
        transaction_costs = 4 * fixed_costs_per_trade + short_costs_per_day

        for day in range(len(spread)):

            delta = pred_spread[day] - spread[day]

            if np.abs(delta) > transaction_costs:  # good to trade
                if delta < 0:  # short spread
                    decision_array[day] = SHORT_SPREAD
                if delta > 0:  # long spread
                    decision_array[day] = LONG_SPREAD
            else:
                decision_array[day] = CLOSE_POSITION

        decision_array[-1] = CLOSE_POSITION

        return decision_array

    def __sax(self, spread_train, spread_full, spread_test, c1_train, c2_train, c1_test, c2_test,
              objectives=["ROI","MDD","SR"],DAYS_CLOSE=252, FIXED_VALUE=1000, commission=0.08,  market_impact=0.2, short_loan=1,
              gen=100, pop=50, w_size=20, alphabet_size=10, verbose=True, plot=False, **kwargs):

        ref_dirs = get_reference_directions(
            "energy", len(objectives), pop, seed=1)
        
        # Build genetic algorithm
        algorithm = NSGA2(pop_size=pop,
                       crossover=PointCrossover(prob=1, n_points=4),
                       mutation=PolynomialMutation(prob=0.1),
                       eliminate_duplicates=True,
                       ref_dirs=ref_dirs)

        # Get objective function
        sax_ga = SaxObjectives(spread=spread_train.to_numpy(), c1=c1_train.to_numpy(), c2=c2_train.to_numpy(), window_size=w_size, alphabet_size=alphabet_size,
                               DAYS_CLOSE=DAYS_CLOSE, FIXED_VALUE=FIXED_VALUE, commission=commission,  market_impact=market_impact, short_loan=short_loan,objectives=objectives)

        # Optimize patterns
        results = minimize(sax_ga, algorithm, ("n_gen", gen),
                           seed=1, save_history=True, verbose=verbose)

        # Define chromossomes intervals
        x = results.X

        MAX_SIZE = w_size
        NON_PATTERN_SIZE = 1+1+1+1
        CHROMOSSOME_SIZE = NON_PATTERN_SIZE+MAX_SIZE
        ENTER_LONG = CHROMOSSOME_SIZE
        ENTER_SHORT = 2*CHROMOSSOME_SIZE

        #create arrays of optimal patterns
        n=len(x)
        dist_long, word_size_long, window_size_long, days_long, pattern_long=np.zeros(n),np.zeros(n,dtype=int),np.zeros(n,dtype=int),np.zeros(n,dtype=int),[]
        dist_short, word_size_short, window_size_short, days_short, pattern_short=np.zeros(n),np.zeros(n,dtype=int),np.zeros(n,dtype=int),np.zeros(n,dtype=int),[]


        # extract chromossomes
        for ind,solution in enumerate(x):

            long_genes = solution[:ENTER_LONG]
            dist_long[ind], word_size_long[ind], window_size_long[ind], days_long[ind], pattern = long_genes[0], round(
                long_genes[1]), round(long_genes[2]), round(long_genes[3]), np.round(long_genes[4:])
            pattern_long.append(pattern[:word_size_long[ind]])


            short_genes = solution[ENTER_LONG:ENTER_SHORT]
            dist_short[ind], word_size_short[ind], window_size_short[ind], days_short[ind], pattern = short_genes[0], round(
                short_genes[1]), round(short_genes[2]), round(long_genes[3]), np.round(short_genes[4:])
            pattern_short.append(pattern[:word_size_short[ind]])



        # From full spread get start of the test set
        spread = spread_full
        i = spread_test.index[0]
        offset = spread_full.index.get_loc(i)

        # Init trade array and trade variables
        trade_array = pd.Series([np.nan for i in range(len(spread_test))])
        trade_array.iloc[0], trade_array.iloc[-1] = CLOSE_POSITION, CLOSE_POSITION
        stabilizing_threshold = 5
        position = CLOSE_POSITION
        l_dist = s_dist = np.inf
        day_count = 0

        for day in range(len(spread_test)-1):

            # Wait for spread to stabilize
            if day < stabilizing_threshold:
                continue

            long_sax_seq, short_sax_seq = get_best_patterns(position=CLOSE_POSITION, spread=spread[:offset+day+1].to_numpy(), alphabet=alphabet_size,
                                                            word_size_long= word_size_long,window_size_long= window_size_long,
                                                            word_size_short= word_size_short,window_size_short= window_size_short)

            # Apply the buy and sell rules
            if position == CLOSE_POSITION:

   
                l_dist,l_idx = get_best_distance(long_sax_seq, pattern_long,dist_long)
                s_dist,s_idx = get_best_distance(short_sax_seq, pattern_short,dist_short)

                # LONG SPREAD
                if l_dist < dist_long[l_idx] and (s_dist >= dist_short[s_idx] or (s_dist < dist_short[s_idx] and l_dist < s_dist)):
                    position, trade_array.iloc[day] = LONG_SPREAD, LONG_SPREAD
                    l_dist = s_dist = np.inf

                elif s_dist < dist_short[s_idx]:  # SHORT SPREAD
                    position, trade_array.iloc[day] = SHORT_SPREAD, SHORT_SPREAD
                    l_dist = s_dist = np.inf

            elif position == LONG_SPREAD:

                if day_count > days_long[l_idx]:
                    position, trade_array.iloc[day] = CLOSE_POSITION, CLOSE_POSITION
                    day_count = 0
                    

            elif position == SHORT_SPREAD:

                if day_count > days_short[s_idx]:
                    position, trade_array.iloc[day] = CLOSE_POSITION, CLOSE_POSITION
                    day_count = 0
   
            if position != CLOSE_POSITION:
                day_count += 1

        # completes the array by propagating the last valid observation
        trade_array = trade_array.fillna(method='ffill')

        if plot:
            plot_positions(spread_test, trade_array)

        return trade_array


    # def __sax(self, spread_train, spread_full, spread_test, c1_train, c2_train, c1_test, c2_test,
    #           objectives=["ROI","MDD","SR"],DAYS_CLOSE=252, FIXED_VALUE=1000, commission=0.08,  market_impact=0.2, short_loan=1,
    #           gen=100, pop=50, w_size=20, alphabet_size=10, verbose=True, plot=False, **kwargs):

    #     ref_dirs = get_reference_directions(
    #         "energy", len(objectives), pop, seed=1)
        
    #     # Build genetic algorithm
    #     algorithm = NSGA2(pop_size=pop,
    #                    crossover=PointCrossover(prob=1, n_points=8),
    #                    mutation=PolynomialMutation(prob=0.1),
    #                    eliminate_duplicates=True,
    #                    ref_dirs=ref_dirs)

    #     # Get objective function
    #     sax_ga = SaxObjectives(spread=spread_train.to_numpy(), c1=c1_train.to_numpy(), c2=c2_train.to_numpy(), window_size=w_size, alphabet_size=alphabet_size,
    #                            DAYS_CLOSE=DAYS_CLOSE, FIXED_VALUE=FIXED_VALUE, commission=commission,  market_impact=market_impact, short_loan=short_loan,objectives=objectives)

    #     # Optimize patterns
    #     results = minimize(sax_ga, algorithm, ("n_gen", gen),
    #                        seed=1, save_history=True, verbose=verbose)

    #     # Define chromossomes intervals
    #     x = results.X

    #     MAX_SIZE = w_size
    #     NON_PATTERN_SIZE = 1+1+1+1
    #     CHROMOSSOME_SIZE = NON_PATTERN_SIZE+MAX_SIZE
    #     ENTER_LONG = CHROMOSSOME_SIZE
    #     EXIT_LONG = 2*CHROMOSSOME_SIZE
    #     ENTER_SHORT = 3*CHROMOSSOME_SIZE
    #     EXIT_SHORT = 4*CHROMOSSOME_SIZE

    #     #create arrays of optimal patterns
    #     n=len(x)
    #     dist_long, word_size_long, window_size_long, days_long, pattern_long=np.zeros(n),np.zeros(n,dtype=int),np.zeros(n,dtype=int),np.zeros(n,dtype=int),[]
    #     dist_exit_long, word_size_exit_long, window_size_exit_long, pattern_exit_long=np.zeros(n),np.zeros(n,dtype=int),np.zeros(n,dtype=int),[]
    #     dist_short, word_size_short, window_size_short, days_short, pattern_short=np.zeros(n),np.zeros(n,dtype=int),np.zeros(n,dtype=int),np.zeros(n,dtype=int),[]
    #     dist_exit_short, word_size_exit_short, window_size_exit_short, pattern_exit_short=np.zeros(n),np.zeros(n,dtype=int),np.zeros(n,dtype=int),[]

    #     # extract chromossomes
    #     for ind,solution in enumerate(x):

    #         long_genes = solution[:ENTER_LONG]
    #         dist_long[ind], word_size_long[ind], window_size_long[ind], days_long[ind], pattern = long_genes[0], round(
    #             long_genes[1]), round(long_genes[2]), round(long_genes[3]), np.round(long_genes[4:])
    #         pattern_long.append(pattern[:word_size_long[ind]])

    #         exit_long_genes = solution[ENTER_LONG:EXIT_LONG]
    #         dist_exit_long[ind], word_size_exit_long[ind], window_size_exit_long[ind], pattern = exit_long_genes[0], round(
    #             exit_long_genes[1]), round(exit_long_genes[2]), np.round(exit_long_genes[4:])
    #         pattern_exit_long.append(pattern[:word_size_exit_long[ind]])

    #         short_genes = solution[EXIT_LONG:ENTER_SHORT]
    #         dist_short[ind], word_size_short[ind], window_size_short[ind], days_short[ind], pattern = short_genes[0], round(
    #             short_genes[1]), round(short_genes[2]), round(long_genes[3]), np.round(short_genes[4:])
    #         pattern_short.append(pattern[:word_size_short[ind]])

    #         exit_short_genes = solution[ENTER_SHORT:EXIT_SHORT]
    #         dist_exit_short[ind], word_size_exit_short[ind], window_size_exit_short[ind], pattern = exit_short_genes[0], round(
    #             exit_short_genes[1]), round(exit_short_genes[2]), np.round(exit_short_genes[4:])
    #         pattern_exit_short.append(pattern[:word_size_exit_short[ind]])

    #     # From full spread get start of the test set
    #     spread = spread_full
    #     i = spread_test.index[0]
    #     offset = spread_full.index.get_loc(i)

    #     # Init trade array and trade variables
    #     trade_array = pd.Series([np.nan for i in range(len(spread_test))])
    #     trade_array.iloc[0], trade_array.iloc[-1] = CLOSE_POSITION, CLOSE_POSITION
    #     stabilizing_threshold = 5
    #     position = CLOSE_POSITION
    #     l_dist = 0
    #     s_dist = 0
    #     day_count = 0

    #     for day in range(len(spread_test)-1):

    #         # Wait for spread to stabilize
    #         if day < stabilizing_threshold:
    #             continue

    #         long_sax_seq, short_sax_seq = get_best_patterns(position, spread[:offset+day+1].to_numpy(), alphabet_size, word_size_long, window_size_long,
    #                                                            word_size_exit_long, window_size_exit_long, word_size_short, window_size_short, word_size_exit_short, window_size_exit_short)

    #         # Apply the buy and sell rules
    #         if position == CLOSE_POSITION:

   
    #             l_dist,l_idx = get_best_distance(long_sax_seq, pattern_long,dist_long)
    #             s_dist,s_idx = get_best_distance(short_sax_seq, pattern_short,dist_short)

    #             # LONG SPREAD
    #             if l_dist < dist_long[l_idx] and (s_dist >= dist_short[s_idx] or (s_dist < dist_short[s_idx] and l_dist < s_dist)):
    #                 position, trade_array.iloc[day] = LONG_SPREAD, LONG_SPREAD
    #                 l_dist = s_dist = 0

    #             elif s_dist < dist_short[s_idx]:  # SHORT SPREAD
    #                 position, trade_array.iloc[day] = SHORT_SPREAD, SHORT_SPREAD
    #                 l_dist = s_dist = 0

    #         elif position == LONG_SPREAD:
    #             l_dist,l_idx = get_best_distance(long_sax_seq, pattern_exit_long,dist_exit_long)
    #             if l_dist > dist_exit_long[l_idx] or day_count > days_long[l_idx]:
    #                 position, trade_array.iloc[day] = CLOSE_POSITION, CLOSE_POSITION
    #                 l_dist = s_dist = np.inf
    #                 day_count = 0

    #         elif position == SHORT_SPREAD:
    #             s_dist,s_idx = get_best_distance(short_sax_seq, pattern_exit_short,dist_exit_short)
    #             if s_dist > dist_exit_short[s_idx] or day_count > days_short[s_idx]:
    #                 position, trade_array.iloc[day] = CLOSE_POSITION, CLOSE_POSITION
    #                 l_dist = s_dist = np.inf
    #                 day_count = 0
    #         if position != CLOSE_POSITION:
    #             day_count += 1

    #     # completes the array by propagating the last valid observation
    #     trade_array = trade_array.fillna(method='ffill')

    #     if plot:
    #         plot_positions(spread_test, trade_array)

    #     return trade_array


    
    # def __sax(self, spread_train, spread_full, spread_test, c1_train, c2_train, c1_test, c2_test,
    #           objectives=["ROI","MDD","SR"],DAYS_CLOSE=252, FIXED_VALUE=1000, commission=0.08,  market_impact=0.2, short_loan=1,
    #           gen=100, pop=50, w_size=20, alphabet_size=10, verbose=True, plot=False, **kwargs):

    #     ref_dirs = get_reference_directions(
    #         "energy", len(objectives), pop, seed=1)
        
    #     # Build genetic algorithm
    #     algorithm = NSGA2(pop_size=pop,
    #                    crossover=PointCrossover(prob=1, n_points=8),
    #                    mutation=PolynomialMutation(prob=0.1),
    #                    eliminate_duplicates=True,
    #                    ref_dirs=ref_dirs)

    #     # Get objective function
    #     sax_ga = SaxObjectives(spread=spread_train.to_numpy(), c1=c1_train.to_numpy(), c2=c2_train.to_numpy(), window_size=w_size, alphabet_size=alphabet_size,
    #                            DAYS_CLOSE=DAYS_CLOSE, FIXED_VALUE=FIXED_VALUE, commission=commission,  market_impact=market_impact, short_loan=short_loan,objectives=objectives)

    #     # Optimize patterns
    #     results = minimize(sax_ga, algorithm, ("n_gen", gen),
    #                        seed=1, save_history=True, verbose=verbose)

    #     # Define chromossomes intervals
    #     x = results.X

    #     MAX_SIZE = w_size
    #     NON_PATTERN_SIZE = 1+1+1+1
    #     CHROMOSSOME_SIZE = NON_PATTERN_SIZE+MAX_SIZE
    #     ENTER_LONG = CHROMOSSOME_SIZE
    #     EXIT_LONG = 2*CHROMOSSOME_SIZE
    #     ENTER_SHORT = 3*CHROMOSSOME_SIZE
    #     EXIT_SHORT = 4*CHROMOSSOME_SIZE

    #     #create arrays of optimal patterns
    #     n=len(x)
    #     dist_long, word_size_long, window_size_long, days_long, pattern_long=np.zeros(n),np.zeros(n,dtype=int),np.zeros(n,dtype=int),np.zeros(n,dtype=int),[]
    #     dist_exit_long, word_size_exit_long, window_size_exit_long, pattern_exit_long=np.zeros(n),np.zeros(n,dtype=int),np.zeros(n,dtype=int),[]
    #     dist_short, word_size_short, window_size_short, days_short, pattern_short=np.zeros(n),np.zeros(n,dtype=int),np.zeros(n,dtype=int),np.zeros(n,dtype=int),[]
    #     dist_exit_short, word_size_exit_short, window_size_exit_short, pattern_exit_short=np.zeros(n),np.zeros(n,dtype=int),np.zeros(n,dtype=int),[]

    #     # extract chromossomes
    #     for ind,solution in enumerate(x):

    #         long_genes = solution[:ENTER_LONG]
    #         dist_long[ind], word_size_long[ind], window_size_long[ind], days_long[ind], pattern = long_genes[0], round(
    #             long_genes[1]), round(long_genes[2]), round(long_genes[3]), np.round(long_genes[4:])
    #         pattern_long.append(pattern[:word_size_long[ind]])

    #         exit_long_genes = solution[ENTER_LONG:EXIT_LONG]
    #         dist_exit_long[ind], word_size_exit_long[ind], window_size_exit_long[ind], pattern = exit_long_genes[0], round(
    #             exit_long_genes[1]), round(exit_long_genes[2]), np.round(exit_long_genes[4:])
    #         pattern_exit_long.append(pattern[:word_size_exit_long[ind]])

    #         short_genes = solution[EXIT_LONG:ENTER_SHORT]
    #         dist_short[ind], word_size_short[ind], window_size_short[ind], days_short[ind], pattern = short_genes[0], round(
    #             short_genes[1]), round(short_genes[2]), round(long_genes[3]), np.round(short_genes[4:])
    #         pattern_short.append(pattern[:word_size_short[ind]])

    #         exit_short_genes = solution[ENTER_SHORT:EXIT_SHORT]
    #         dist_exit_short[ind], word_size_exit_short[ind], window_size_exit_short[ind], pattern = exit_short_genes[0], round(
    #             exit_short_genes[1]), round(exit_short_genes[2]), np.round(exit_short_genes[4:])
    #         pattern_exit_short.append(pattern[:word_size_exit_short[ind]])

    #     # From full spread get start of the test set
    #     spread = spread_full
    #     i = spread_test.index[0]
    #     offset = spread_full.index.get_loc(i)

    #     # Init trade array and trade variables
    #     trade_array = pd.Series([np.nan for i in range(len(spread_test))])
    #     trade_array.iloc[0], trade_array.iloc[-1] = CLOSE_POSITION, CLOSE_POSITION
    #     stabilizing_threshold = 5
    #     position = CLOSE_POSITION
    #     l_dist = 0
    #     s_dist = 0
    #     day_count = 0

    #     for day in range(len(spread_test)-1):

    #         # Wait for spread to stabilize
    #         if day < stabilizing_threshold:
    #             continue

    #         long_sax_seq, short_sax_seq = get_best_patterns(position, spread[:offset+day+1].to_numpy(), alphabet_size, word_size_long, window_size_long,
    #                                                            word_size_exit_long, window_size_exit_long, word_size_short, window_size_short, word_size_exit_short, window_size_exit_short)

    #         # Apply the buy and sell rules
    #         if position == CLOSE_POSITION:

   
    #             l_dist,l_idx = get_best_distance(long_sax_seq, pattern_long,dist_long)
    #             s_dist,s_idx = get_best_distance(short_sax_seq, pattern_short,dist_short)

    #             # LONG SPREAD
    #             if l_dist < dist_long[l_idx] and (s_dist >= dist_short[s_idx] or (s_dist < dist_short[s_idx] and l_dist < s_dist)):
    #                 position, trade_array.iloc[day] = LONG_SPREAD, LONG_SPREAD
    #                 l_dist = s_dist = 0

    #             elif s_dist < dist_short[s_idx]:  # SHORT SPREAD
    #                 position, trade_array.iloc[day] = SHORT_SPREAD, SHORT_SPREAD
    #                 l_dist = s_dist = 0

    #         elif position == LONG_SPREAD:
    #             l_dist,l_idx = get_best_distance(long_sax_seq, pattern_exit_long,dist_exit_long)
    #             if l_dist > dist_exit_long[l_idx] or day_count > days_long[l_idx]:
    #                 position, trade_array.iloc[day] = CLOSE_POSITION, CLOSE_POSITION
    #                 l_dist = s_dist = np.inf
    #                 day_count = 0

    #         elif position == SHORT_SPREAD:
    #             s_dist,s_idx = get_best_distance(short_sax_seq, pattern_exit_short,dist_exit_short)
    #             if s_dist > dist_exit_short[s_idx] or day_count > days_short[s_idx]:
    #                 position, trade_array.iloc[day] = CLOSE_POSITION, CLOSE_POSITION
    #                 l_dist = s_dist = np.inf
    #                 day_count = 0
    #         if position != CLOSE_POSITION:
    #             day_count += 1

    #     # completes the array by propagating the last valid observation
    #     trade_array = trade_array.fillna(method='ffill')

    #     if plot:
    #         plot_positions(spread_test, trade_array)

    #     return trade_array

    def __sax_ga(self, spread_train, spread_full, spread_test, c1_train, c2_train, c1_test, c2_test,
              DAYS_CLOSE=252, FIXED_VALUE=1000, commission=0.08,  market_impact=0.2, short_loan=1,
              gen=100, pop=50, w_size=20, alphabet_size=10, verbose=True, plot=False, **kwargs):
        # Build genetic algorithm
        algorithm = GA(pop_size=pop,
                       crossover=PointCrossover(prob=1, n_points=8),
                       mutation=PolynomialMutation(prob=0.1),
                       eliminate_duplicates=True)

        # Get objective function
        sax_ga = SaxObjectivesGA(spread=spread_train.to_numpy(), c1=c1_train.to_numpy(
        ), c2=c2_train.to_numpy(), window_size=w_size,alphabet_size=alphabet_size,DAYS_CLOSE=DAYS_CLOSE,FIXED_VALUE=FIXED_VALUE,commission=commission,  market_impact=market_impact, short_loan=short_loan)

        # Optimize patterns
        results = minimize(sax_ga, algorithm, ("n_gen", gen),
                           seed=1, save_history=True, verbose=verbose)

        # Define chromossomes intervals
        x = results.X

        MAX_SIZE = w_size
        NON_PATTERN_SIZE = 1+1+1+1
        CHROMOSSOME_SIZE = NON_PATTERN_SIZE+MAX_SIZE
        ENTER_LONG = CHROMOSSOME_SIZE
        EXIT_LONG = 2*CHROMOSSOME_SIZE
        ENTER_SHORT = 3*CHROMOSSOME_SIZE
        EXIT_SHORT = 4*CHROMOSSOME_SIZE

        # extract chromossomes
        long_genes = x[:ENTER_LONG]
        dist_long, word_size_long, window_size_long, days_long, pattern_long = long_genes[0], round(
            long_genes[1]), round(long_genes[2]), round(long_genes[3]), np.round(long_genes[4:])
        pattern_long = pattern_long[:word_size_long]

        exit_long_genes = x[ENTER_LONG:EXIT_LONG]
        dist_exit_long, word_size_exit_long, window_size_exit_long, pattern_exit_long = exit_long_genes[0], round(
            exit_long_genes[1]), round(exit_long_genes[2]), np.round(exit_long_genes[4:])
        pattern_exit_long = pattern_exit_long[:word_size_exit_long]

        short_genes = x[EXIT_LONG:ENTER_SHORT]
        dist_short, word_size_short, window_size_short, days_short, pattern_short = short_genes[0], round(
            short_genes[1]), round(short_genes[2]), round(long_genes[3]), np.round(short_genes[4:])
        pattern_short = pattern_short[:word_size_short]

        exit_short_genes = x[ENTER_SHORT:EXIT_SHORT]
        dist_exit_short, word_size_exit_short, window_size_exit_short, pattern_exit_short = exit_short_genes[0], round(
            exit_short_genes[1]), round(exit_short_genes[2]), np.round(exit_short_genes[4:])
        pattern_exit_short = pattern_exit_short[:word_size_exit_short]

        # From full spread get start of the test set
        spread = spread_full
        i = spread_test.index[0]
        offset = spread_full.index.get_loc(i)

        # Init trade array and trade variables
        trade_array = pd.Series([np.nan for i in range(len(spread_test))])
        trade_array.iloc[0], trade_array.iloc[-1] = CLOSE_POSITION, CLOSE_POSITION
        stabilizing_threshold = 5
        position = CLOSE_POSITION
        l_dist = 0
        s_dist = 0
        day_count = 0

        for day in range(len(spread_test)-1):

            # Wait for spread to stabilize
            if day < stabilizing_threshold:
                continue

            long_sax_seq, short_sax_seq = sax_ga._get_patterns(position, spread[:offset+day+1].to_numpy(), alphabet_size, word_size_long, window_size_long,
                                                               word_size_exit_long, window_size_exit_long, word_size_short, window_size_short, word_size_exit_short, window_size_exit_short)

            # Apply the buy and sell rules
            if position == CLOSE_POSITION:

                if long_sax_seq is not None:
                    l_dist = pattern_distance(long_sax_seq, pattern_long)
                if short_sax_seq is not None:
                    s_dist = pattern_distance(short_sax_seq, pattern_short)

                # LONG SPREAD
                if l_dist < dist_long and (s_dist >= dist_short or (s_dist < dist_short and l_dist < s_dist)):
                    position, trade_array.iloc[day] = LONG_SPREAD, LONG_SPREAD
                    l_dist = s_dist = 0

                elif s_dist < dist_short:  # SHORT SPREAD
                    position, trade_array.iloc[day] = SHORT_SPREAD, SHORT_SPREAD
                    l_dist = s_dist = 0

            elif position == LONG_SPREAD:
                if long_sax_seq is not None:
                    l_dist = pattern_distance(long_sax_seq, pattern_exit_long)
                if l_dist > dist_exit_long or day_count > days_long:
                    position, trade_array.iloc[day] = CLOSE_POSITION, CLOSE_POSITION
                    l_dist = s_dist = np.inf
                    day_count = 0

            elif position == SHORT_SPREAD:
                if short_sax_seq is not None:
                    s_dist = pattern_distance(
                        short_sax_seq, pattern_exit_short)
                if s_dist > dist_exit_short or day_count > days_short:
                    position, trade_array.iloc[day] = CLOSE_POSITION, CLOSE_POSITION
                    l_dist = s_dist = np.inf
                    day_count = 0
            if position != CLOSE_POSITION:
                day_count += 1

        # completes the array by propagating the last valid observation
        trade_array = trade_array.fillna(method='ffill')

        if plot:
            plot_positions(spread_test, trade_array)

        return trade_array

    def run_simulation(self, model):

        # Select function
        function = {'TH': self.__threshold, 'SAX': self.__sax,
                    'FA': self.__forecasting_algorithm}

        # Select function arguments
        args = load_args(model)

        # Get price series / data tickers / pairs identified
        data = self.__data
        tickers = self.__tickers
        pairs = self.__pairs

        # Initialize trade variables
        n_pairs = len(pairs)
        n_non_convergent_pairs = 0
        profit = 0
        loss = 0
        profit_loss_trade = np.zeros(2)
        total_trades = 0

        # Fixed_value is based on last simulation portfolio value
        FIXED_VALUE = self.__INIT_VALUE / n_pairs

        for component1, component2 in pairs:  # get components for each pair

            # Extract tickers in each component
            component1 = [(ticker in component1) for ticker in tickers]
            component2 = [(ticker in component2) for ticker in tickers]

            # Get one series for each component
            c1 = price_of_entire_component(data, component1)
            c2 = price_of_entire_component(data, component2)

            # Get series between train/test/full date intervals
            c1_train = dataframe_interval(
                self.__train_start, self.__train_end, c1)
            c2_train = dataframe_interval(
                self.__train_start, self.__train_end, c2)
            c1_test = dataframe_interval(
                self.__test_start, self.__test_end, c1)
            c2_test = dataframe_interval(
                self.__test_start, self.__test_end, c2)
            c1_full = dataframe_interval(
                self.__train_start, self.__test_end, c1)
            c2_full = dataframe_interval(
                self.__train_start, self.__test_end, c2)

            # Get beta coefficient and spread for train/test/full
            beta, spread_train = coint_spread(c1_train, c2_train)
            spread_full = c1_full-beta*c2_full
            spread_test = c1_test-beta*c2_test

            # Apply trading model and get trading decision array
            trade_array = function[model](spread_train=spread_train, spread_full=spread_full,
                                          spread_test=spread_test, c1_train=c1_train, c2_train=c2_train, c1_test=c1_test, c2_test=c2_test, **args)

            # Force close non convergent positions
            trade_array = self.__force_close(trade_array)

            # Apply trading rules to trade decision array
            n_trades, cash, portfolio_value, days_open, profitable_unprofitable = self.__trade_spread(
                c1=c1_test, c2=c2_test, trade_array=trade_array, FIXED_VALUE=FIXED_VALUE, **load_args("TRADING"))

            # Evaluate pair performance
            pair_performance = portfolio_value[-1]/portfolio_value[0] * 100
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
            "cash": list(aux_cash),
            "profit_pairs": profit,
            "loss_pairs": loss,
            "profit_trades": int(profit_loss_trade[0]),
            "loss_trades": int(profit_loss_trade[1]),
            "non_convergent_pairs": n_non_convergent_pairs
        }

        # Change portfolio init value for next simulation
        self.__INIT_VALUE = aux_pt_value[-1]

        return stats
