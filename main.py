
from modules.pair_formation import PairFormation
from modules.trading_phase import TradingPhase
from modules.portfolio import Portfolio
from utils.utils import date_change,get_data,save_pickle,tuple_int,argparse

def main(pairs_alg,trading_alg,index,sector,start_date,end_date,months_trading,months_forming):

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
        selected_pairs=pair_formation.find_pairs(pairs_alg,verbose=False,plot=False)

        trading_phase.set_pairs(selected_pairs["pairs"])
        trading_phase.set_dates(train_start,train_end,test_start,test_end)
        performance=trading_phase.run_simulation(trading_alg,verbose=False,plot=False)

        portfolio.report(selected_pairs,performance,verbose=False)

        train_start=date_change(train_start,months_trading)
        train_end=date_change(train_end,months_trading)
        test_start=date_change(test_start,months_trading)
        test_end=date_change(test_end,months_trading)

    portfolio.evaluate()

    save_pickle(portfolio)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--index',type=str,default='s&p500')
    parser.add_argument('--sector',type=str,default='Real Estate')
    parser.add_argument('--start_date',type=tuple_int,default=(2015,1,1))
    parser.add_argument('--end_date',type=tuple_int,default=(2022,1,1))
    parser.add_argument('--months_trading',type=int,default=12)
    parser.add_argument('--months_forming',type=int,default=36)
    parser.add_argument('--pairs_alg',type=str,default='DIST')
    parser.add_argument('--trading_alg',type=str,default='TH')

    args=parser.parse_args()

    main(args.pairs_alg,args.trading_alg,args.index,args.sector,args.start_date,args.end_date,args.months_trading,args.months_forming)
