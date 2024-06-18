

import numpy as np

from operator import itemgetter

from statsmodels.tsa.stattools import coint
from hurst import compute_Hc as hurst_exponent

from utils.utils import date_string, dataframe_interval, load_args,coint_spread,get_membership
from os.path import exists
import pickle

class PairFormation:

    """
    A class used to represent PairFormation models

    """

    def __init__(self, data,sector,membership_date,membership_sector):

        self.__all_pairs = []  # list of pairs
        self.__entire_data = data  # complete price series
        # price series (to be used for date intervals)
        self.__data = self.__entire_data
        self.__tickers = data.keys()  # data tickers
        # start date of data (will be discarded for train start)
        self.__start = data.index[0]
        # end date of data (will be discarded for train end)
        self.__end = data.index[-1]
        self.__sector = sector if isinstance(sector, list) else [sector]
        self.__membership_date = membership_date  # list of pairs
        self.__membership_sector = membership_sector  # list of pairs

    def set_date(self, train_start, train_end,test_start,test_end):

        # Set tickers and dates for train
        self.__start = date_string(train_start)
        self.__end = date_string(train_end)
        data=dataframe_interval(self.__start, date_string(test_end), self.__entire_data)
        clean_data=data.dropna(axis=1).loc[:, ~(data < 0).any()]
        membership=get_membership(clean_data,self.__membership_date,date_string(test_start))
        self.__data = dataframe_interval(self.__start, self.__end, membership)
        self.__tickers = self.__data.keys()

        
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

    def __cointegrated_pairs(self, pvalue_threshold=0.05, hurst_threshold=0.5 ,**kwargs):

        compute_pair=False
        file='results/'+'engle_granger.pkl'
        id = self.__start+'_'+self.__end+'_'+str(pvalue_threshold)
        isExist = exists(file)

        sectors=self.__membership_sector.keys() if self.__sector[0] == "All" else self.__sector
        
        if isExist:
            # Load the data from a pickle file
            with open(file, 'rb') as f:
                pairs = pickle.load(f)
        else:
            pairs={}

        res=[]

        for sector in sectors:

            string=id+f'_{sector}'

            if compute_pair or (string not in pairs):  
                # Initialize price series and tickers
                data = self.__data.filter(items=self.__membership_sector[sector])
                tickers = data.columns
                N = len(tickers)
                p=[]
                
                # for each combination of tickers appends pairs if coint coef and hurst are below thresholds
                for i in range(N):
                    signal1 = data[tickers[i]]
                    for j in range(i+1, N):
                        signal2 = data[tickers[j]]

                        t_stat, p_value, crit_value = coint(np.log(signal1), np.log(signal2))

                        if p_value >= pvalue_threshold:
                            continue
                        # Compute the spread
                        beta,spread,_,_=coint_spread(signal1, signal2)

                        hurst,c,d=hurst_exponent(spread)
                        if beta<0:
                            continue
                        if hurst>=hurst_threshold:
                            continue
                        p.append([[tickers[i]], [tickers[j]]])
                pairs.update({string:np.array(p)})
                # Save the array to a pickle file
                with open(file, 'wb') as f:
                    pickle.dump(pairs, f)
            
            res.extend(pairs[string])                

        self.__all_pairs = res


    def find_pairs(self, model):

        # Select function
        function = {'COINT': self.__cointegrated_pairs,
                    'DIST': self.__distance_pairs}

        # Apply pair formation model and find pairs
        function[model](**load_args(model))

        # PairFormation dictionary
        stats = {"model": model,
                "formation_start": self.__start,
                "formation_end":  self.__end,
                "n_tickers": len(self.__data.columns),
                "n_pairs": len(self.__all_pairs),
                "pairs": self.__all_pairs,
                }

        return stats
