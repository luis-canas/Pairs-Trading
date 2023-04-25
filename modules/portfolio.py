
import pprint
from utils.utils import date_string


class Portfolio:

    """
    A class used to represent PairsTrading information

    """
    
    def __init__(self, data, index, sector, start_date, end_date, months_trading, months_forming, pairs_alg, trading_alg):

        self.data = data  # price series
        self.index = index  # market index
        self.sector = sector  # market sector
        self.tickers = data.keys()  # data ticker
        self.start_date = date_string(start_date)  # simulation start date
        self.end_date = date_string(end_date)  # simulation end date
        # training period (pair_formation)
        self.months_trading = months_trading
        self.months_forming = months_forming  # testing period (trading_phase)
        self.pairs_alg = pairs_alg  # pair formation algorithm
        self.trading_alg = trading_alg  # trading phase algorithm
        self.pair_info = []  # pair formation list (of dicts)
        self.portfolio_info = []  # trading phase list (of dicts)

    def report(self, pairs, performance):

        # add current simulation pairs and portfolio performance to dict
        self.pair_info.append(pairs)
        self.portfolio_info.append(performance)

    def evaluate(self, verbose=False):

        pass

    def print_pairs(self, simul=-1):

        if simul == -1:  # no simulation given or -1 prints, prints pairs for all simul
            for simul in range(len(self.pair_info)):
                print('formation_start: ',
                      self.pair_info[simul]['formation_start'])
                print(self.pair_info[simul]['pairs'])
                print('\n')

        elif simul < len(self.pair_info):  # prints pairs for given simulation
            print('formation_start: ',
                  self.pair_info[simul]['formation_start'])
            print(self.pair_info[simul]['pairs'])

        else:  # invalid simulation given
            print('Invalid simulation')

    def print(self):

        # print simulations for pair_formation/trading_phase
        for simul in range(len(self.portfolio_info)):
            pprint.pprint(
                [self.pair_info[simul], self.portfolio_info[simul]], depth=2, sort_dicts=False)
