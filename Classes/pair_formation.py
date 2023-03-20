

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from operator import itemgetter
from statsmodels.api import OLS
from statsmodels.tsa.stattools import coint, adfuller
from hurst import compute_Hc as hurst_exponent
from scipy.stats import zscore

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

from pymoo.util.ref_dirs import get_reference_directions


import statsmodels.api as sm
from tqdm import tqdm
from pymoo.core.problem import ElementwiseProblem


from utils import *
# Constants
NB_TRADING_DAYS = 252  # 1 year has 252 trading days

# the number of constraints is always the same (since ADF is always an objective - ADF is the default objective)
N_CONSTRAINTS = 6  # 4 regarding the number of elements, 1 since a stocks can't be in both components at the same time, 1 regarding coint cutoff
"""MIN_ELEMENTS = 2  # min and max stocks per component
MAX_ELEMENTS = 5"""
COINT_CUTOFF = 0.05  # the adf test has a maximum value so that a time series is can be considered stationary



class Objectives(ElementwiseProblem):
    
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

class PairFormation:

    """
    A class used to represent trading Pairs

    """

    def __init__(self, data):

        
        self.__all_pairs = []
        self.__entire_data = data
        self.__data = data
        self.__tickers = data.keys()
        self.__start = data.index[0]
        self.__end = data.index[-1]

    def set_date(self,start,end):

        self.__start = date_string(start)
        self.__end = date_string(end)
        self.__data = dataframe_interval(self.__start, self.__end,self.__entire_data)
        self.__tickers = self.__data.keys()


    def __nsga2(self,verbose,plot):


        gen = 80
        objective_functions = ['ADF', 'spread_volatility', 'NZC', 'half_life']

        ref_dirs = get_reference_directions("energy", len(objective_functions), 50, seed=1)

        algorithm = NSGA2(pop_size=50,
                        sampling=BinaryRandomSampling(),
                        crossover=TwoPointCrossover(),
                        mutation=BitflipMutation(),
                        eliminate_duplicates=True, 
                        ref_dirs=ref_dirs)


        pairs_objectives = Objectives(self.__data, self.__tickers, objective_functions, min_elements = 1, max_elements = 1)

        results = minimize(pairs_objectives, algorithm, ("n_gen", gen), seed=1, save_history=True, verbose=verbose)
        
        if plot:
            study_results(results, objective_functions, gen)
        
        self.__all_pairs = results_to_tickers(results, self.__tickers)



    def __distance_pairs(self,pair_number =  20,verbose=False,plot=False):

        data = self.__data
        tickers = self.__tickers
        N = len(tickers)

        dic = {}

        
        for i in tqdm(range(N)):

            signal1 = data[tickers[i]]

            for j in range(i+1, N):

                signal2 = data[tickers[j]]
                
                ssd=sum((np.array(signal1) - np.array(signal2))**2)

                
                dic[tickers[i], tickers[j]]=ssd


        top_pairs = list(dict(sorted(dic.items(), key = itemgetter(1))[:pair_number]).keys())

        self.__all_pairs = [[[x], [y]] for x, y in top_pairs]

    def __is_stationary(self,signal, threshold):

        signal=np.asfarray(signal)
        return True if adfuller(signal)[1] < threshold else False  

    def __cointegrated_pairs(self, pvalue_threshold = 0.05,hurst_threshold = 0.5,verbose=False,plot=False):

        data = self.__data
        tickers = self.__tickers
        N = len(tickers)

        pairs = []

        for i in tqdm(range(N)):

            signal1 = data[tickers[i]]

            for j in range(i+1, N):

                signal2 = data[tickers[j]]
                
                if self.__Engle_Granger(signal1, signal2, pvalue_threshold, hurst_threshold,plot):
                    pairs.append([[tickers[i]], [tickers[j]]])

        self.__all_pairs = pairs




    def __Engle_Granger(self, signal1, signal2, pvalue_threshold=0.05, hurst_threshold=0.5,plot=False):

        beta = OLS(signal2, signal1).fit().params[0]
        spread = signal2-beta*signal1
        result = coint(signal1, signal2)
        score = result[0]
        pvalue = result[1]
        hurst, _, _ = hurst_exponent(spread)
  
        if(plot and pvalue <= pvalue_threshold and hurst <= hurst_threshold):
            plt.figure(figsize=(12, 6))
            normalized_spread = zscore(spread)
            normalized_spread.plot()
            standard_devitation = np.std(normalized_spread)
            plt.axhline(zscore(spread).mean())
            plt.axhline(standard_devitation, color='green')
            plt.axhline(-standard_devitation, color='green')
            plt.axhline(2*standard_devitation, color='red')
            plt.axhline(-2*standard_devitation, color='red')
            plt.xlim(self.__start,
                     self.__end)
            plt.show()

        return True if pvalue <= pvalue_threshold and hurst <= hurst_threshold else False

    def find_pairs(self,model,verbose=False,plot=False):

        function = {'COINT':self.__cointegrated_pairs,'DIST':self.__distance_pairs,'NSGA':self.__nsga2}
        function[model](verbose=verbose,plot=plot)

        info={"model": model,
                "formation_start": self.__start,
                "formation_end":  self.__end,
                "n_tickers": len(self.__tickers),
                "n_pairs": len(self.__all_pairs),
                "n_unique_tickers":len(np.unique(self.__all_pairs)),
                "pairs":self.__all_pairs,
        }
                    
        return info





