
from class_Pairs import Pairs
from class_Trader import Trader
from class_Portfolio import Portfolio
from utils import dataframe_interval,compute_pca,get_data

def main():

    pairs_alg='COINT'
    trading_alg='TH'
    index='s&p500'
    # sector='Financials'
    sector='Real Estate'

    start_date=(2015,1,1)
    end_date=(2020,1,1)


    data=get_data(index,sector,start_date,end_date)

    selector=Pairs(data)
    strategy=Trader(data)



    #SPLIT
    train_start=(2015,1,1)
    train_end=(2016,1,1)
    test_start=(2016,1,1)
    test_end=(2017,1,1)
    
    # Find Tickers
    selector.set_date(train_start,train_end)
    selected_pairs=selector.find_pairs(pairs_alg,verbose=True,plot=False)

    # Test Tickers
    strategy.set_pairs(selected_pairs)
    strategy.set_dates(train_start,train_end,test_start,test_end)
    strategy.run_simulation(trading_alg,verbose=False,plot=False)



if __name__ == "__main__":
    main()