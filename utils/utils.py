import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import statsmodels.api as sm
import random
from os import makedirs
from os.path import isfile, exists
import yfinance as yf
import pickle
import json

file_screener = 'screeners/'
file_input = 'results/'
file_output = 'results/'
file_args="modules/args.json"


def save_pickle(item):

    isExist = exists(file_output)
    if not isExist:
        makedirs(file_output)

    name = item.index+'_'+item.sector+'_'+item.start_date+'_'+item.end_date+'_' + \
        str(item.months_forming)+'_'+str(item.months_trading) + \
        '_'+item.pairs_alg+'_'+item.trading_alg

    with open(file_output+name+'.pkl', 'wb') as output:
        pickle.dump(item, output, pickle.HIGHEST_PROTOCOL)


def open_pickle(pairs_alg, trading_alg, index, sector, start_date, end_date, months_trading, months_forming):

    name = index+'_'+sector+'_'+date_string(start_date)+'_'+date_string(end_date)+'_'+str(
        months_forming)+'_'+str(months_trading)+'_'+pairs_alg+'_'+trading_alg

    isExist = exists(file_output+name+'.pkl')

    if not isExist:
        print('Portfolio ', name, ' does not exist!')

        return
    else:
        with open(file_output+name+'.pkl', 'rb') as input:
            portfolio = pickle.load(input)

        return portfolio


def study_results(res, objectives, n_gen):
    X, F = res.opt.get("X", "F")

    hist = res.history
    print(len(hist))

    n_evals = []  # corresponding number of function evaluations\
    hist_F = []  # the objective space values in each generation
    hist_cv = []  # constraint violation in each generation
    hist_cv_avg = []  # average constraint violation in the whole population

    for algo in hist:
        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)

        # retrieve the optimum from the algorithm
        opt = algo.opt

        # store the least contraint violation and the average in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())

        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])

    k = np.where(np.array(hist_cv) <= 0.0)[0].min()
    print(
        f"At least one feasible solution in Generation {k} after {n_evals[k]} evaluations.")
    vals = hist_cv_avg
    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, vals, color='black', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, vals, facecolor="none", edgecolor='black', marker="p")
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Constraint Violation")
    plt.legend()
    plt.show()

    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)

    from pymoo.indicators.hv import Hypervolume

    if len(objectives) == 2:
        ref = np.array([1, 1])
    elif len(objectives) == 4:
        ref = np.array([1, 1, 1, 1])

    metric = Hypervolume(ref_point=ref,
                         norm_ref_point=False,
                         zero_to_one=True,
                         ideal=approx_ideal,
                         nadir=approx_nadir)

    hv = [metric.do(_F) for _F in hist_F]

    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv, color='black', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, hv, facecolor="none", edgecolor='black', marker="p")
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.show()

    from pymoo.util.running_metric import RunningMetricAnimation

    running = RunningMetricAnimation(delta_gen=n_gen,
                                     n_plots=1,
                                     key_press=False,
                                     do_show=True)

    for algorithm in res.history:
        running.update(algorithm)

    if len(objectives) == 2:
        plt.figure(figsize=(7, 5))
        plt.scatter(F[:, 0], F[:, 1], s=30,
                    facecolors='none', edgecolors='blue')
        plt.title("Objective Space")
        plt.xlabel(objectives[0])
        plt.ylabel(objectives[1])
        plt.show()
    elif len(objectives) == 4:  # plottar 3 eixos é o melhor que se pode fazer
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(F[:, 2], F[:, 1], F[:, 3], zdir='z',
                   s=30, c=None, depthshade=True)
        ax.set_xlabel(objectives[2])
        ax.set_ylabel(objectives[1])
        ax.set_zlabel(objectives[3])
        plt.title("Objective Space")
        plt.show()


def results_to_tickers(res, tickers):

    selected_stocks = [[]
                       for _ in range(len(res.X) * 2)]  # two components per pair

    counter = 0

    for result in res.X:
        for i in range(len(result)):
            if result[i]:
                if i < len(result) // 2:
                    selected_stocks[counter].append(tickers[i])
                else:
                    selected_stocks[counter +
                                    1].append(tickers[i - (len(result) // 2)])

        counter += 2

    pairs = [[selected_stocks[i], selected_stocks[i + 1]]
             for i in range(0, len(selected_stocks), 2)]

    return pairs


def dataframe_interval(start_date, end_date, data):

    mask = (data.index > start_date) & (data.index <= end_date)

    return data.loc[mask]


def date_string(date):

    return datetime(*date).strftime("%Y-%m-%d")


def date_change(date, timeframe):

    year, month, day = date[0], date[1], date[2]

    year = year + (month + timeframe - 1) // 12
    month = (month + timeframe - 1) % 12 + 1

    newdate = (year, month, day)

    return newdate


def coint_spread(c1, c2):
    S1 = np.asarray(c1)
    S2 = np.asarray(c2)
    S1_c = sm.add_constant(S1)

    results = sm.OLS(S1_c, S2).fit()

    b = results.params[0][1]

    coint_spread = c1 - b*c2

    return b, coint_spread


def compute_zscore(full_spread, test_spread):

    i = test_spread.index[0]
    offset = full_spread.index.get_loc(i)

    norm_spread = np.zeros(len(test_spread))

    mean = np.zeros(len(test_spread))
    std = np.zeros(len(test_spread))
    t_spread = np.zeros(len(test_spread))

    for day, daily_value in enumerate(test_spread):
        spread_to_consider = full_spread[(day+1): (day+1) + offset]

        norm_spread[day] = (
            daily_value - spread_to_consider.mean()) / spread_to_consider.std()

        mean[day] = spread_to_consider.mean()
        std[day] = spread_to_consider.std()
        t_spread[day] = daily_value

    return norm_spread, mean, std, t_spread


def price_of_entire_component(series, component):

    if not any(component):
        # just to return something, this subject will be discarded for not having enough stocks in the component
        return series.iloc[:, random.randint(0, 10)]

    combined_series = series.iloc[:, component].sum(axis=1)

    return combined_series


def get_data(index, sector, start, end):

    isExist = exists(file_input)
    if not isExist:
        makedirs(file_input)

    if not isfile(file_input+f'{index}_{sector}_{date_string(start)}_{date_string(end)}.csv'):
        df = pd.read_csv(
            file_screener+f'{index}_screener.csv', encoding='latin1')

        mask = df['Sector'].str.contains(sector)
        mask = mask.where(pd.notnull(mask), False).tolist()
        tickers = df['Symbol']
        tickers = tickers[mask].tolist()

        data = yf.download(tickers, start=datetime(
            *start), end=datetime(*end))['Close']

        data.to_csv(
            file_input+f'{index}_{sector}_{date_string(start)}_{date_string(end)}.csv')
    else:
        data = pd.read_csv(
            file_input+f'{index}_{sector}_{date_string(start)}_{date_string(end)}.csv', index_col='Date')

    return data

    # start = datetime(*start)
    # end = datetime(*end)

    # tickers = ['AAPL', 'ADBE', 'ORCL', 'EBAY', 'MSFT', 'QCOM', 'HPQ', 'JNPR', 'AMD', 'IBM', 'SPY']

    # return yf.download(tickers, start, end)['Close']


def tuple_int(string):
    try:
        x, y, z = map(int, string.split(","))
        return (x, y, z)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid input format. Please provide comma separated integers.")


def load_args(model):

    with open(file_args, "r") as f:
        data = json.load(f)

    # Extract the args from the dictionary
    return data[model]

def change_args(model,parameter,newvalue):

    # Open JSON file in read mode
    with open(file_args, 'r') as f:
        data = json.load(f)

    # Modify the value of a key in the dictionary
    try:
        data[model][parameter] = newvalue
    except KeyError:
        print("Model/parameter does not exist")

    # Open the same file in write mode
    with open(file_args, 'w') as f:
        # Write the modified dictionary to the file
        json.dump(data, f,indent=4)

def max_drawdown(s, verbose = False):
  i = np.argmax(np.maximum.accumulate(s) - s) # end of the period
  j = np.argmax(s[:i]) # start of period

  mdd = ((s[j] - s[i])/s[j] ) * 100

  if verbose:
    plt.plot(s)
    plt.plot([i, j], [s[i], s[j]], 'o', color='Red', markersize=10)
    plt.show()

  return mdd, i, j
