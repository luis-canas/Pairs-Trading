
from class_Pairs import Pairs
from class_Trader import Trader
from class_History import History


def main():

    portfolios = {'DEFAULT'}
    pairs_algs={'COINT','DIST'}
    trading_algs={'MA','ARMA'}

    start_date=(2013,1,1)
    end_date=(2018,1,1)

    # Get Price History
    series=History()
    data=series.get_data('DEFAULT',start=start_date,end=end_date)


    #SPLIT
    

    # Find Tickers
    selector=Pairs(data)
    selected_pairs=selector.find_pairs('COINT',verbose=True)

    # Test Tickers
    strategy=Trader(data)
    strategy.set_pairs(selected_pairs)
    strategy.run_simulation('MA',verbose=True)



if __name__ == "__main__":
    main()