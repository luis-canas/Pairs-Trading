
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from utils.utils import price_of_entire_component
from pymoo.core.problem import ElementwiseProblem
# Constants
NB_TRADING_DAYS = 252  # 1 year has 252 trading days

# the number of constraints is always the same (since ADF is always an objective - ADF is the default objective)
N_CONSTRAINTS = 6  # 4 regarding the number of elements, 1 since a stocks can't be in both components at the same time, 1 regarding coint cutoff
"""MIN_ELEMENTS = 2  # min and max stocks per component
MAX_ELEMENTS = 5"""
COINT_CUTOFF = 0.05  # the adf test has a maximum value so that a time series is can be considered stationary




class PairFormationObjectives(ElementwiseProblem):
    
    def __init__(self, price_series, tickers, objective_functions, min_elements=1, max_elements=1):

        self.N_VARIABLES = 2 * len(tickers)  # 1 decision variable per stock, times 2 since a pair has 2 components

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
        if 'ADF' in self.objectives: # ADF should always be in the objetives actually
            """NAO DEVIA SER COINT(C1, C2)????? DEVIA DEVIA"""
            _, f1, _ = coint(c2_series, c1_series)  # minimize to ensure that the series are cointegrated
            aux_obj.append(f1)
        if 'spread_volatility' in self.objectives:
            f2 = -self.__volatility_test(spread)  # maximizing the spread's volatility means that it is as dynamic as possible
            aux_obj.append(f2)
        if 'NZC' in self.objectives:
            f3 = -self.__zero_crossings(spread)  # maximizing the NZC means that the spread is dynamic and mean reverting
            aux_obj.append(f3)
        if 'half_life' in self.objectives:
            f4 = self.__half_life(spread)  # min. half life ensures that the spread converges quickly
            aux_obj.append(f4)
        # constraints
        g1 = np.dot(component_1, ones) - self.MAX_ELEMENTS  # both components cannot have more than MAX_ELEMENTS
        g2 = np.dot(component_2, ones) - self.MAX_ELEMENTS
        g3 = - np.dot(component_1, ones) + self.MIN_ELEMENTS  # both components have to have at least MIN_ELEMENTS
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



    def __zero_crossings(self,x):
        """
        Function that counts the number of zero crossings of a given signal
        :param x: the signal to be analyzed
        """
        x = x - x.mean()
        x_arr = np.array(x)
        nzc = sum(1 for i, _ in enumerate(x_arr) if (i + 1 < len(x_arr)) if ((x_arr[i] * x_arr[i + 1] < 0) or (x_arr[i] == 0)))

        return nzc
        
    def __coint_spread(self,s2, s1):
        S1 = np.asarray(s1)
        S2 = np.asarray(s2)

        S1_c = sm.add_constant(S1)

        results = sm.OLS(S2, S1_c).fit()
        b = results.params[1]

        spread = s2 - b * s1

        return spread

    def __half_life(self,z_array):
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

    def __volatility_test(self,spread):
        """
        This function returns the HISTORICAL VOLATILITY of the spread
        :param spread: spread between the 2 components defined as s = c1 - beta*c2
        :return: HISTORICAL VOLATILITY of the spread
        """

        return spread.std()