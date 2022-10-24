
from class_Pairs import Pairs
from class_Trader import Trader
from class_History import History


def main():

    portfolios = {'DEFAULT','PICKLE'}
    pairs_algs={'COINT','DIST'}
    trading_algs={'MA','ARMA','TH'}

    start_date=(2013,1,1)
    end_date=(2018,1,1)

    # Get Price History
    series=History()
    series.set_date(*start_date,*end_date)
    data=series.get_data('DEFAULT')


    #SPLIT
    
    # Find Tickers
    selector=Pairs(data)
    selected_pairs=selector.find_pairs('COINT',verbose=False)

    # Test Tickers
    strategy=Trader(data)
    strategy.set_pairs(selected_pairs)
    strategy.run_simulation('TH',verbose=True)



if __name__ == "__main__":
    main()