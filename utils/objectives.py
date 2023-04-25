
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from utils.symbolic_aggregate_approximation import timeseries2symbol, min_dist, pattern_distance, find_pattern
from utils.utils import price_of_entire_component
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.pntx import TwoPointCrossover


# Constants
NB_TRADING_DAYS = 252  # 1 year has 252 trading days

# the number of constraints is always the same (since ADF is always an objective - ADF is the default objective)
N_CONSTRAINTS = 6  # 4 regarding the number of elements, 1 since a stocks can't be in both components at the same time, 1 regarding coint cutoff
"""MIN_ELEMENTS = 2  # min and max stocks per component
MAX_ELEMENTS = 5"""
COINT_CUTOFF = 0.05  # the adf test has a maximum value so that a time series is can be considered stationary

LONG_SPREAD = 1
SHORT_SPREAD = -1
CLOSE_POSITION = 0


class PairFormationObjectives(ElementwiseProblem):

    def __init__(self, price_series, tickers, objective_functions, min_elements=1, max_elements=1):

        # 1 decision variable per stock, times 2 since a pair has 2 components
        self.N_VARIABLES = 2 * len(tickers)

        self.tickers = tickers

        self.MIN_ELEMENTS = min_elements
        self.MAX_ELEMENTS = max_elements

        self.objectives = objective_functions

        self.N_OBJECTIVES = len(self.objectives)

        self.price_series = price_series

        super().__init__(n_var=self.N_VARIABLES,
                         n_obj=self.N_OBJECTIVES,
                         n_constr=N_CONSTRAINTS,
                         xl=0,  # binary variable
                         xu=1,  # binary variable
                         typer_var=int)  # only assigns integers to the variables

    def _evaluate(self, x, out, *args, **kwargs):

        aux_obj = []

        component_1 = [bool(boolean) for boolean in x[:len(x) // 2]]
        component_2 = [bool(boolean) for boolean in x[len(x) // 2:]]
        ones = np.ones(len(x) // 2)  # vector of 1's used in the constraints

        # creates the artificial time series given by the combination of stocks in component 1, 2
        c1_series = price_of_entire_component(self.price_series, component_1)
        c2_series = price_of_entire_component(self.price_series, component_2)

        # calculates the spread given by s = c1 - Beta*c2, where Beta is the cointegration factor
        spread = self.__coint_spread(c1_series, c2_series)

        # objective functions
        if 'ADF' in self.objectives:  # ADF should always be in the objetives actually
            """NAO DEVIA SER COINT(C1, C2)????? DEVIA DEVIA"""
            _, f1, _ = coint(
                c2_series, c1_series)  # minimize to ensure that the series are cointegrated
            aux_obj.append(f1)
        if 'spread_volatility' in self.objectives:
            # maximizing the spread's volatility means that it is as dynamic as possible
            f2 = -self.__volatility_test(spread)
            aux_obj.append(f2)
        if 'NZC' in self.objectives:
            # maximizing the NZC means that the spread is dynamic and mean reverting
            f3 = -self.__zero_crossings(spread)
            aux_obj.append(f3)
        if 'half_life' in self.objectives:
            # min. half life ensures that the spread converges quickly
            f4 = self.__half_life(spread)
            aux_obj.append(f4)
        # constraints
        # both components cannot have more than MAX_ELEMENTS
        g1 = np.dot(component_1, ones) - self.MAX_ELEMENTS
        g2 = np.dot(component_2, ones) - self.MAX_ELEMENTS
        # both components have to have at least MIN_ELEMENTS
        g3 = - np.dot(component_1, ones) + self.MIN_ELEMENTS
        g4 = - np.dot(component_2, ones) + self.MIN_ELEMENTS

        # a stock can not belong to both components
        for i, j in zip(component_1, component_2):
            if i and j:
                g5 = 1  # invalid
                break
            else:
                g5 = -1  # valid value

        g6 = f1 - COINT_CUTOFF  # the EG test of the spread has to be lower than a certain cutoff

        # pymoo requires that objective functions and constraints are gathered in the variable out
        out["G"] = [g1, g2, g3, g4, g5, g6]
        out["F"] = [f for f in aux_obj]

    def __zero_crossings(self, x):
        """
        Function that counts the number of zero crossings of a given signal
        :param x: the signal to be analyzed
        """
        x = x - x.mean()
        x_arr = np.array(x)
        nzc = sum(1 for i, _ in enumerate(x_arr) if (i + 1 < len(x_arr))
                  if ((x_arr[i] * x_arr[i + 1] < 0) or (x_arr[i] == 0)))

        return nzc

    def __coint_spread(self, s2, s1):
        S1 = np.asarray(s1)
        S2 = np.asarray(s2)

        S1_c = sm.add_constant(S1)

        results = sm.OLS(S2, S1_c).fit()
        b = results.params[1]

        spread = s2 - b * s1

        return spread

    def __half_life(self, z_array):
        """
        This function calculates the half life parameter of a
        mean reversion series
        """
        z_array = np.array(z_array)
        z_lag = np.roll(z_array, 1)
        z_lag[0] = 0
        z_ret = z_array - z_lag
        z_ret[0] = 0

        # adds intercept terms to X variable for regression
        z_lag2 = sm.add_constant(z_lag)

        model = sm.OLS(z_ret[1:], z_lag2[1:])
        res = model.fit()

        halflife = -np.log(2) / res.params[1]

        return halflife

    def __volatility_test(self, spread):
        """
        This function returns the HISTORICAL VOLATILITY of the spread
        :param spread: spread between the 2 components defined as s = c1 - beta*c2
        :return: HISTORICAL VOLATILITY of the spread
        """

        return spread.std()


class SaxObjectives(ElementwiseProblem):

    def __init__(self, spread, c1, c2, window_size, word_size, alphabet_size):

        # spread and components price series
        self.spread = spread
        self.c1 = c1
        self.c2 = c2

        # sax parameters
        self.window_size = window_size
        self.word_size = word_size
        self.alphabet_size = alphabet_size

        # Chromossome size: pattern_distance for decision and pattern of decision
        CHROMOSSOME_SIZE = 1+word_size

        # Chromossome for each decision (4 strategies)
        self.ENTER_LONG = CHROMOSSOME_SIZE
        self.EXIT_LONG = 2*CHROMOSSOME_SIZE
        self.ENTER_SHORT = 3*CHROMOSSOME_SIZE
        self.EXIT_SHORT = 4*CHROMOSSOME_SIZE

        # lower bound and upper bound for pattern_distances
        variables_lb = [0]
        variables_ub = [100]

        # lower bound and upper bound for patterns
        pattern_lb = [0]*word_size
        pattern_ub = [alphabet_size-1]*word_size

        # join bounds
        x1 = np.tile(np.concatenate((variables_lb, pattern_lb)), 4)
        xu = np.tile(np.concatenate((variables_ub, pattern_ub)), 4)

        super().__init__(n_var=4*CHROMOSSOME_SIZE,
                         n_obj=1,
                         n_constr=0,
                         xl=x1,
                         xu=xu,
                         vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):

        # extract chromossomes
        long_genes = x[:self.ENTER_LONG]
        dist_long, pattern_long = long_genes[0], np.round(long_genes[1:])

        exit_long_genes = x[self.ENTER_LONG:self.EXIT_LONG]
        dist_exit_long, pattern_exit_long = exit_long_genes[0], np.round(
            exit_long_genes[1:])

        short_genes = x[self.EXIT_LONG:self.ENTER_SHORT]
        dist_short, pattern_short = short_genes[0], np.round(short_genes[1:])

        exit_short_genes = x[self.ENTER_SHORT:self.EXIT_SHORT]
        dist_exit_short, pattern_exit_short = exit_short_genes[0], np.round(
            exit_short_genes[1:])

        # Initialize variables for tracking trades and earnings
        in_position = False
        position = CLOSE_POSITION
        FIXED_VALUE = 1000
        stocks_in_hand = np.zeros(2)
        cash_in_hand = FIXED_VALUE
        l_dist = 0
        s_dist = 0

        # Slide a window along the time series and convert to SAX
        for i in range(len(self.spread) - self.window_size):

            day = i+self.window_size
            window = self.spread[i:day]
            sax_seq, _ = find_pattern(
                window, self.word_size, self.alphabet_size)

            # Apply the buy and sell rules
            if not in_position:

                l_dist = pattern_distance(sax_seq, pattern_long)
                s_dist = pattern_distance(sax_seq, pattern_short)

                if l_dist < dist_long:  # LONG SPREAD

                    in_position, position = True, LONG_SPREAD

                    value_to_buy = min(FIXED_VALUE, cash_in_hand)
                    # long c1
                    cash_in_hand += -value_to_buy
                    stocks_in_hand[0] = value_to_buy / self.c1[day]
                    # short c2
                    cash_in_hand += value_to_buy
                    stocks_in_hand[1] = -value_to_buy / self.c2[day]

                elif s_dist < dist_short:  # SHORT SPREAD

                    in_position, position = True, SHORT_SPREAD

                    value_to_buy = min(FIXED_VALUE, cash_in_hand)
                    # long c2
                    cash_in_hand += -value_to_buy
                    stocks_in_hand[1] = value_to_buy / self.c2[day]
                    # short c1
                    cash_in_hand += value_to_buy
                    stocks_in_hand[0] = -value_to_buy / self.c1[day]

            elif in_position:

                if position == LONG_SPREAD:
                    l_dist = pattern_distance(sax_seq, pattern_exit_long)
                elif position == SHORT_SPREAD:
                    s_dist = pattern_distance(sax_seq, pattern_exit_short)

                if (l_dist > dist_exit_long and position == LONG_SPREAD) or (s_dist > dist_exit_short and position == SHORT_SPREAD):

                    in_position, position = False, CLOSE_POSITION
                    sale_value = stocks_in_hand[0] * \
                        self.c1[day] + stocks_in_hand[1] * self.c2[day]
                    cash_in_hand += sale_value
                    # both positions were closed
                    stocks_in_hand[0] = stocks_in_hand[1] = 0

        portfolio_value = cash_in_hand + \
            stocks_in_hand[0] * self.c1[day] + stocks_in_hand[1] * self.c2[day]

        # Set the fitness value to the total earnings
        out["F"] = -portfolio_value
