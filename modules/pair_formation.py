

import numpy as np

from operator import itemgetter

from statsmodels.tsa.stattools import coint
from hurst import compute_Hc as hurst_exponent


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

from pymoo.util.ref_dirs import get_reference_directions

from utils.genetic_algorithm import PairFormationObjectives
from utils.utils import date_string, dataframe_interval, study_results, results_to_tickers, load_args,coint_spread


class PairFormation:

    """
    A class used to represent PairFormation models

    """

    def __init__(self, data):

        self.__all_pairs = []  # list of pairs
        self.__entire_data = data.dropna(axis=1)  # complete price series
        # price series (to be used for date intervals)
        self.__data = self.__entire_data
        self.__tickers = data.keys()  # data tickers
        # start date of data (will be discarded for train start)
        self.__start = data.index[0]
        # end date of data (will be discarded for train end)
        self.__end = data.index[-1]

    def set_date(self, start, end):

        # Set tickers and dates for train
        self.__start = date_string(start)
        self.__end = date_string(end)
        self.__data = dataframe_interval(
            self.__start, self.__end, self.__entire_data)
        self.__tickers = self.__data.keys()

    def __nsga2(self, gen=80, pop=50, objective_functions=['ADF', 'spread_volatility', 'NZC', 'half_life'], min_elements=1, max_elements=1, verbose=True, plot=False, **kwargs):

        ref_dirs = get_reference_directions(
            "energy", len(objective_functions), pop, seed=1)

        # Build NSGA2
        algorithm = NSGA2(pop_size=pop,
                          sampling=BinaryRandomSampling(),
                          crossover=TwoPointCrossover(),
                          mutation=BitflipMutation(),
                          eliminate_duplicates=True,
                          ref_dirs=ref_dirs)

        # Get pair formation objectives - ['ADF', 'spread_volatility', 'NZC', 'half_life']
        pairs_objectives = PairFormationObjectives(
            self.__data, self.__tickers, objective_functions, min_elements, max_elements)

        # Optimize function
        results = minimize(pairs_objectives, algorithm, ("n_gen", gen),
                           seed=1, save_history=True, verbose=verbose)

        if plot:
            study_results(results, objective_functions, gen)

        # Extract pairs
        self.__all_pairs = results_to_tickers(results, self.__tickers)

    def __distance_pairs(self, pair_number=5, **kwargs):

        # Initialize price series and tickers
        data = self.__data
        tickers = self.__tickers
        N = len(tickers)

        dic = {}

        # for each combination of tickers in dataset calculate ssd and add to dict
        for i in range(N):

            signal1 = data[tickers[i]]

            for j in range(i+1, N):

                signal2 = data[tickers[j]]

                ssd = sum((np.array(signal1) - np.array(signal2))**2)

                dic[tickers[i], tickers[j]] = ssd

        # order dict by ssd and reduce to pair_number size
        top_pairs = list(
            dict(sorted(dic.items(), key=itemgetter(1))[:pair_number]).keys())

        # create list for top pairs
        self.__all_pairs = [[[x], [y]] for x, y in top_pairs]

    def __cointegrated_pairs(self, pvalue_threshold=0.05, hurst_threshold=0.5, **kwargs):

        # Initialize price series and tickers
        data = self.__data
        tickers = self.__tickers
        N = len(tickers)

        pairs = []

        # for each combination of tickers appends pairs if coint coef and hurst are below thresholds
        for i in range(N):

            signal1 = data[tickers[i]]

            for j in range(i+1, N):

                signal2 = data[tickers[j]]

                t_stat, p_value, crit_value = coint(signal1, signal2)
                
                if p_value >= pvalue_threshold:
                    continue
            
                
                # Compute the Hurst exponent for spread
                beta,spread=coint_spread(signal1, signal2)
                H, _ ,_= hurst_exponent(spread)


                if H >= hurst_threshold :
                    continue

                pairs.append([[tickers[i]], [tickers[j]]])

        self.__all_pairs = pairs

    def find_pairs(self, model):

        # Select function
        function = {'COINT': self.__cointegrated_pairs,
                    'DIST': self.__distance_pairs, 'NSGA': self.__nsga2}

        # Apply pair formation model and find pairs
        function[model](**load_args(model))

        # PairFormation dictionary
        stats = {"model": model,
                "formation_start": self.__start,
                "formation_end":  self.__end,
                "n_tickers": len(self.__tickers),
                "n_pairs": len(self.__all_pairs),
                "pairs": self.__all_pairs,
                }

        return stats
