
from class_Pairs import Pairs
from class_Trader import Trader
from class_History import History
from utils import dataframe_interval,compute_pca

def main():

    portfolios = {'DEFAULT','PICKLE'}
    pairs_algs={'COINT','DIST','NSGA'}
    trading_algs={'MA','ARMA','TH'}

    start_date=(2017,1,1)
    end_date=(2018,1,1)

    # Get Price History
    series=History()
    series.set_date(start_date,end_date)
    data=series.get_data('DEFAULT')
    data=dataframe_interval(start_date,end_date,data)

    data=compute_pca(11,data.T).T


    #SPLIT



    # Find Tickers
    selector=Pairs(data)
    selected_pairs=selector.find_pairs('NSGA',verbose=True,plot=True)

    # Test Tickers
    strategy=Trader(data)
    strategy.set_pairs(selected_pairs)
    strategy.run_simulation('TH',verbose=False)



if __name__ == "__main__":
    main()