
from class_Pairs import Pairs
from class_Trader import Trader
from class_Portfolio import Portfolio
from utils import *

def main():

    pairs_alg='DIST'
    trading_alg='TH'
    index='s&p500'
    sector='Financials'
    # sector='Real Estate'

    start_date=(2015,1,1)
    end_date=(2020,1,1)
    train_start=(2015,1,1)
    train_end=(2016,1,1)
    test_start=(2016,1,1)
    test_end=(2017,1,1)
    months_inc=12
    n_simul=1

    data=get_data(index,sector,start_date,end_date)
    pair_formation=Pairs(data)
    trading_phase=Trader(data)
    portfolio=Portfolio(data,index,sector,start_date,end_date,months_inc)

    
    for year_index in range(n_simul):

        pair_formation.set_date(train_start,train_end)
        selected_pairs=pair_formation.find_pairs(pairs_alg,verbose=True,plot=False)


        trading_phase.set_pairs(selected_pairs["pairs"])
        trading_phase.set_dates(train_start,train_end,test_start,test_end)
        performance=trading_phase.run_simulation(trading_alg,verbose=False,plot=False)

        portfolio.report(selected_pairs,performance,verbose=True)

        train_start=date_change(train_start,months_inc)
        train_end=date_change(train_start,months_inc)
        test_start=date_change(train_start,months_inc)
        test_end=date_change(train_start,months_inc)





if __name__ == "__main__":
    main()