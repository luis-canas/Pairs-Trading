from class_Pairs import Pairs
from class_Trader import Trader
from class_Portfolio import Portfolio


def main():

    portfolio=Portfolio()
    portfolio.portfolio1()
    data_train,data_val=portfolio.get_train_val()


    # Find Tickers
    find_pairs=Pairs(data_train)
    find_pairs.cointegrated_pairs()
    selected_pairs=find_pairs.get_pairs()
    print(selected_pairs)

    # Validate Tickers
    trader=Trader(data_train)
    trader.set_pairs(selected_pairs)
    trader.run_simulation('MA')


if __name__ == "__main__":
    main()