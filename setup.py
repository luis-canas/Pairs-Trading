
from class_Pairs import Pairs
from class_Trader import Trader
from class_History import History
from utils import dataframe_interval,compute_pca

def main():

    portfolios = {'DEFAULT','PICKLE'}
    pairs_algs={'COINT','DIST','NSGA'}
    trading_algs={'MA','ARMA','TH'}

    start_date=(2015,1,1)
    end_date=(2020,1,1)

    # Get Price History
    series=History()
    series.set_date(start_date,end_date)
    data=series.get_data('PORTFOLIO')


    #SPLIT
    train_start=(2015,1,1)
    train_end=(2016,1,1)
    test_start=start_date=(2016,1,1)
    test_end=(2017,1,1)
    # Find Tickers
    selector=Pairs(data)
    selected_pairs=selector.find_pairs('DIST',verbose=False,plot=False)

    # Test Tickers
    strategy=Trader(data)
    strategy.set_pairs(selected_pairs)
    strategy.set_dates(train_start,train_end,test_start,test_end)
    strategy.run_simulation('TH',verbose=False,plot=False)



if __name__ == "__main__":
    main()