
import pprint
from utils.utils import date_string


class Portfolio:

    def __init__(self, data, index, sector, start_date, end_date, months_trading, months_forming, pairs_alg, trading_alg):

        self.data = data
        self.index = index
        self.sector = sector
        self.tickers = data.keys()
        self.start_date = date_string(start_date)
        self.end_date = date_string(end_date)
        self.months_trading = months_trading
        self.months_forming = months_forming
        self.pairs_alg = pairs_alg
        self.trading_alg = trading_alg
        self.pair_info = []
        self.portfolio_info = []

    def report(self, pairs, performance):

        self.pair_info.append(pairs)
        self.portfolio_info.append(performance)

    def evaluate(self, verbose=False):

        pass

    def print_pairs(self, year=-1):

        if year == -1:
            for simul in range(len(self.pair_info)):
                print('formation_start: ',
                      self.pair_info[simul]['formation_start'])
                print(self.pair_info[simul]['pairs'])
                print('\n')
        elif year < len(self.pair_info):
            print('formation_start: ', self.pair_info[year]['formation_start'])
            print(self.pair_info[year]['pairs'])
        else:
            print('Invalid year')

    def plot_pairs(self, pair, year):
        pass

    def print(self):

        for simul in range(len(self.portfolio_info)):
            pprint.pprint(
                [self.pair_info[simul], self.portfolio_info[simul]], depth=2, sort_dicts=False)
