

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

from utils.objectives import PairFormationObjectives
from utils.utils import date_string,dataframe_interval,study_results,results_to_tickers



class PairFormation:

    """
    A class used to represent trading Pairs

    """

    def __init__(self, data):

        
        self.__all_pairs = []
        self.__entire_data = data.dropna(axis=1)
        self.__data = self.__entire_data
        self.__tickers = data.keys()
        self.__start = data.index[0]
        self.__end = data.index[-1]

    def set_date(self,start,end):

        self.__start = date_string(start)
        self.__end = date_string(end)
        self.__data=dataframe_interval(self.__start, self.__end,self.__entire_data)
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


        pairs_objectives = PairFormationObjectives(self.__data, self.__tickers, objective_functions, min_elements = 1, max_elements = 1)

        results = minimize(pairs_objectives, algorithm, ("n_gen", gen), seed=1, save_history=True, verbose=verbose)
        
        if plot:
            study_results(results, objective_functions, gen)
        
        self.__all_pairs = results_to_tickers(results, self.__tickers)



    def __distance_pairs(self,pair_number =  5,verbose=False,plot=False):

        data = self.__data
        tickers = self.__tickers
        N = len(tickers)

        dic = {}

        
        for i in range(N):

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

        for i in range(N):

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



