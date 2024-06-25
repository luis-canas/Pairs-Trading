
from modules.pair_formation import PairFormation
from modules.trading_phase import TradingPhase
from modules.portfolio import Portfolio
from utils.utils import date_change, get_data, save_pickle, tuple_int, argparse

def main(pairs_alg, trading_alg, index, sector, start_date, end_date, months_trading, months_forming):

    # Calculate initial date variables based on arguments
    train_start = start_date
    train_end = date_change(train_start, months_forming)
    test_start = train_end
    test_end = date_change(test_start, months_trading)

    # Calculate the number of months between start_date and end_date
    start_year, start_month, start_day = start_date
    end_year, end_month, end_day = end_date
    num_months = (end_year - start_year) * 12 + end_month - start_month

    # Calculate the number of simulations
    num_simulations = (num_months - months_forming - months_trading) // months_trading + 1

    # Get historical data and init modules
    data,membership_date,membership_sector = get_data(index)
    # data = get_data(index, sector, start_date, end_date)
    pair_formation = PairFormation(data,sector,membership_date,membership_sector)
    trading_phase = TradingPhase(data,sector)
    portfolio = Portfolio(data, index, sector, start_date, end_date,membership_sector,membership_date,
                          months_trading, months_forming, pairs_alg, trading_alg)

    for _ in range(num_simulations):

        # Set training interval and get pairs
        pair_formation.set_date(train_start, train_end, test_start, test_end)
        selected_pairs = pair_formation.find_pairs(pairs_alg)

        # Set training/test interval and pairs
        # Trade pairs
        trading_phase.set_pairs(selected_pairs["pairs"])
        trading_phase.set_dates(train_start, train_end, test_start, test_end)

        performance = trading_phase.run_simulation(trading_alg)

        # Save pairs and performance dictionaries for current simulation
        portfolio.report(selected_pairs, performance)

        # Calculate next date variables based on arguments
        train_start = date_change(train_start, months_trading)
        train_end = date_change(train_end, months_trading)
        test_start = date_change(test_start, months_trading)
        test_end = date_change(test_end, months_trading)


    # Evaluate portfolio after simulations and get performance metrics
    portfolio.evaluate()
    portfolio.model(trading_phase.model_performance(trading_alg))

    # Save portfolio to .pkl
    save_pickle(portfolio)


if __name__ == "__main__":

    # Get arguments from command line

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--index', type=str, default='s&p500')
    parser.add_argument('--sector', type=str, default='All')
    parser.add_argument('--start_date', type=tuple_int, default=(2017, 1, 1))
    parser.add_argument('--end_date', type=tuple_int, default=(2023, 1, 1))
    parser.add_argument('--months_trading', type=int, default=12)
    parser.add_argument('--months_forming', type=int, default=60)
    parser.add_argument('--pairs_alg', type=str, default='COINT')
    parser.add_argument('--trading_alg', type=str, default='TH')
    args = parser.parse_args()

    main(args.pairs_alg, args.trading_alg, args.index, args.sector,
         args.start_date, args.end_date, args.months_trading, args.months_forming)
