

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.api import OLS
from statsmodels.tsa.stattools import coint, adfuller
from hurst import compute_Hc as hurst_exponent
from scipy.stats import zscore

from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

VERBOSE = False


class Pairs:

    """
    A class used to represent trading Pairs

    """

    def __init__(self, data):

        self.all_pairs = []
        self.data = data
        self.tickers = data.keys()
        self.start = data.index[0]._date_repr
        self.end = data.index[-1]._date_repr

    def is_stationary(signal, threshold):
        return True if adfuller(signal)[1] < threshold else False

    def get_pairs(self):
        return self.all_pairs

    def cointegrated_pairs(self):

        data = self.data
        tickers = self.tickers
        n_pairs = len(tickers)

        adfuller_threshold = 1
        pvalue_threshold = 0.05
        hurst_threshold = 0.5  # mean reversing threshold

        pairs = []

        for i in range(n_pairs):

            pair1 = data[tickers[i]]
            if (not self.is_stationary(pair1, adfuller_threshold)):
                continue

            for j in range(i+1, n_pairs):

                pair2 = data[tickers[j]]
                if(not self.is_stationary(pair2, adfuller_threshold)):
                    continue

                _, pvalue, hurst = self.Engle_Granger(
                    pair1, pair2, pvalue_threshold, hurst_threshold)

                if pvalue <= pvalue_threshold and hurst <= hurst_threshold:
                    pairs.append((tickers[i], tickers[j]))

        self.all_pairs = pairs

    def Engle_Granger(self, signal1, signal2, pvalue_threshold, hurst_threshold):

        beta = OLS(signal2, signal1).fit().params[0]
        spread = signal2-beta*signal1
        result = coint(signal1, signal2)
        score = result[0]
        pvalue = result[1]
        hurst, _, _ = hurst_exponent(spread)

        if(VERBOSE and pvalue <= pvalue_threshold and hurst <= hurst_threshold):
            plt.figure(figsize=(12, 6))
            normalized_spread = zscore(spread)
            normalized_spread.plot()
            standard_devitation = np.std(normalized_spread)
            plt.axhline(zscore(spread).mean())
            plt.axhline(standard_devitation, color='green')
            plt.axhline(-standard_devitation, color='green')
            plt.axhline(2*standard_devitation, color='red')
            plt.axhline(-2*standard_devitation, color='red')
            plt.xlim(self.start,
                     self.end)
            plt.show()

        return score, pvalue, hurst

    # def compute_PCA(self, n_components=0.1):

    #     scaler = MinMaxScaler()
    #     data_rescaled = scaler.fit_transform(self.data.T)

    #     pca = PCA(n_components=2)
    #     pca.fit(data_rescaled)

    #     reduced = pca.transform(data_rescaled)

    #     self.yi=reduced

    #     return reduced

    # def compute_OPTICS(self):

    #     clf = OPTICS()
    #     print(clf)

    #     clf.fit(self.yi)
    #     labels = clf.labels_
    #     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #     print("Clusters discovered: %d" % n_clusters_)

    #     clustered_series_all = pd.Series(index=self.data.columns, data=labels.flatten())
    #     clustered_series = clustered_series_all[clustered_series_all != -1]

    #     counts = clustered_series.value_counts()
    #     print("Pairs to evaluate: %d" % (counts * (counts - 1) / 2).sum())
