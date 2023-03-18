
from pairformation import PairFormation
from tradingphase import TradingPhase
from portfolio import Portfolio
from utils import *

def main(pairs_alg,trading_alg,index,sector,start_date,end_date,months_trading,months_forming):

    #define simulation parameters
    train_start=start_date
    train_end=date_change(train_start,months_forming)
    test_start=train_end
    test_end=date_change(test_start,months_trading)
    years_simulated=(end_date[0] - test_end[0])+1

    data=get_data(index,sector,start_date,end_date)
    pair_formation=PairFormation(data)
    trading_phase=TradingPhase(data)
    portfolio=Portfolio(data,index,sector,start_date,end_date,months_trading,months_forming,pairs_alg,trading_alg)

    
    for _ in range(years_simulated):

        pair_formation.set_date(train_start,train_end)
        selected_pairs=pair_formation.find_pairs(pairs_alg,verbose=True,plot=False)

        trading_phase.set_pairs(selected_pairs["pairs"])
        trading_phase.set_dates(train_start,train_end,test_start,test_end)
        performance=trading_phase.run_simulation(trading_alg,verbose=False,plot=False)

        portfolio.report(selected_pairs,performance,verbose=True)

        train_start=date_change(train_start,months_trading)
        train_end=date_change(train_end,months_trading)
        test_start=date_change(test_start,months_trading)
        test_end=date_change(test_end,months_trading)

    portfolio.evaluate()

    save_pickle(portfolio)




if __name__ == "__main__":

    # pairs_alg='DIST'
    # trading_alg='TH'
    # index='s&p500'
    # sector='Financials'
    # # sector='Real Estate'


    # #simulation initial parameters
    # start_date=(2015,1,1)
    # end_date=(2020,1,1)
    # months_trading=12
    # months_forming=12

    pairs_alg='NSGA'
    trading_alg='TH'
    index='s&p500'
    # sector='Financials'
    sector='Real Estate'


    #simulation initial parameters
    start_date=(2015,1,1)
    end_date=(2022,1,1)
    months_trading=12
    months_forming=12*3

    open_pickle(pairs_alg,trading_alg,index,sector,start_date,end_date,months_trading,months_forming)

    # main(pairs_alg,trading_alg,index,sector,start_date,end_date,months_trading,months_forming)