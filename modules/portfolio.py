
import pprint
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import date_string,max_drawdown,sharpe_ratio


class Portfolio:

    """
    A class used to represent PairsTrading information

    """
    
    def __init__(self, data, index, sector, start_date, end_date,membership_sector, membership_date, months_trading, months_forming, pairs_alg, trading_alg):

        self.data = data  # price series
        self.index = index  # market index
        self.sector = sector  # market sector
        self.tickers = data.keys()  # data ticker
        self.start_date = date_string(start_date)  # simulation start date
        self.end_date = date_string(end_date)  # simulation end date
        self.membership_sector = membership_sector
        self.membership_date = membership_date
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

    def evaluate(self):

        total_portfolio_value=[]
        total_cash=[]
        self.evaluation={}
        max_mdd=0
        roi_avg=0
        sr_avg=0
        profit_pairs=0
        pprofit_pairs=0
        total_pairs=0
        profit_trades=0
        loss_trades=0
        mdd_avg=0
        cointegrated_pairs=0
        for simulation in self.portfolio_info:
            total_portfolio_value+=simulation['portfolio_value']
            total_cash+=simulation['cash']
            
            roi = (simulation['portfolio_value'][-1]/(simulation['portfolio_value'][0])-1) * 100
            simulation_cash=np.array(simulation['portfolio_value'])
            mdd=max_drawdown(simulation_cash)[0]
            sr=sharpe_ratio(simulation_cash,simulation["trading_start"])
            self.evaluation[simulation["trading_start"]+'/'+simulation["trading_end"]]={}
            self.evaluation[simulation["trading_start"]+'/'+simulation["trading_end"]]['roi']=roi
            self.evaluation[simulation["trading_start"]+'/'+simulation["trading_end"]]['mdd']=mdd
            self.evaluation[simulation["trading_start"]+'/'+simulation["trading_end"]]['sr']=sr
            if mdd>max_mdd:
                max_mdd=mdd
            roi_avg+=roi
            sr_avg+=sr
            mdd_avg+=mdd
            profit_pairs+=simulation['profit_pairs']
            total_pairs+=simulation['n_pairs']
            pprofit_pairs+=simulation['profit_pairs']/simulation['n_pairs']
            profit_trades+=simulation['profit_trades']
            loss_trades+=simulation['loss_trades']
            cointegrated_pairs+=simulation['cointegrated_pairs']

        total_portfolio_value=np.array(total_portfolio_value)
        roi = (total_portfolio_value[-1]/(total_portfolio_value[0])-1) * 100
        roi_avg/=len(self.portfolio_info)
        sr_avg/=len(self.portfolio_info)
        mdd_avg/=len(self.portfolio_info)
        self.pairs_avg=total_pairs/len(self.portfolio_info)
        # profit_pairs_avg=pprofit_pairs/len(self.portfolio_info)
        profit_trades_avg=profit_trades/len(self.portfolio_info)
        self.break_coint_pairs=(1-cointegrated_pairs/total_pairs)*100
        loss_trades_avg=loss_trades/len(self.portfolio_info)
        self.total_portfolio_value=total_portfolio_value
        self.profit_pairs=pprofit_pairs/len(self.portfolio_info)
        self.evaluation["coint_pairs"]=cointegrated_pairs/total_pairs*100
        self.evaluation["total_roi"]=roi
        self.evaluation["roi_avg"]=roi_avg
        self.evaluation["sr_avg"]=sr_avg
        self.evaluation["mdd_avg"]=mdd_avg



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
                [self.portfolio_info[simul]], depth=2, sort_dicts=False)
    def performance(self):

        # print metrics
        pprint.pprint(
            [self.evaluation], sort_dicts=False,width=50)
        
    def plot_sectors(self):
        # Initialize a dictionary to store the counts
        sector_counts = {sector: 0 for sector in self.membership_sector.keys()}

        # Iterate over each simulation
        for simul in self.pair_info:
            # Iterate over each pair in the simulation
            for pair in simul['pairs']:
                sector=1
                try:
                    for sec, tickers in self.membership_sector.items():
                        if (pair[0][0] in tickers) or (pair[1][0] in tickers):
                            sector=sec
                            break
                except:
                    pass
                if sector==1: 
                    continue

                sector_counts[sector]+=1

        print(sector_counts)
        # Mapping of your keys to the S&P 500 sector names
        key_mapping = {
            'Industrials': 'Industrials',
            'Healthcare': 'Health Care',
            'Technology': 'Information Technology',
            'Consumer Cyclical': 'Consumer Discretionary',
            'Financial Services': 'Financials',
            'Consumer Defensive': 'Consumer Staples',
            'Utilities': 'Utilities',
            'Real Estate': 'Real Estate',
            'Energy': 'Energy',
            'Basic Materials': 'Materials',
            'Communication Services': 'Communication Services'
        }

        # Create a new dictionary with the updated keys
        data = {key_mapping[key]: value for key, value in sector_counts.items()}
        labels = data.keys()
        values = data.values()

       # Calculate the total count
        total = sum(data.values())

        # Calculate the percentage for each sector
        percentages = {key: (value / total) * 100 for key, value in data.items()}


        # Sort the dictionary by value
        percentages = dict(sorted(percentages.items(), key=lambda item: item[1], reverse=True))

        fig, ax = plt.subplots()

        # Create a custom color palette with as many colors as there are sectors
        colors = sns.color_palette('Blues_r', len(data))  # 'Blues_r' is a reversed Blues palette

        wedges, texts = ax.pie(percentages.values(), startangle=90, colors=colors)
        sns.set_context("paper", font_scale = 2)

        # Create legend with percentages
        legend_labels = [f'{k} - {v:.1f}%' for k, v in percentages.items()]

        # Adjust the bbox_to_anchor values
        plt.legend(wedges, legend_labels, title="Sectors", loc="center left", bbox_to_anchor=(0.75, 0, 0.5, 1))

        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.show()
