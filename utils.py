import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from sklearn import preprocessing
from sklearn.decomposition import PCA

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
    print(f"At least one feasible solution in Generation {k} after {n_evals[k]} evaluations.")
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
        plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
        plt.title("Objective Space")
        plt.xlabel(objectives[0])
        plt.ylabel(objectives[1])
        plt.show()
    elif len(objectives) == 4:  # plottar 3 eixos é o melhor que se pode fazer
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(F[:, 2], F[:, 1], F[:, 3], zdir='z', s=30, c=None, depthshade=True)
        ax.set_xlabel(objectives[2])
        ax.set_ylabel(objectives[1])
        ax.set_zlabel(objectives[3])
        plt.title("Objective Space")
        plt.show()




def results_to_tickers(res, tickers):

    pairs = []

    for result in res.X:
        for i in range(len(result)):
            if result[i]:
                if i < len(result) // 2:
                    signal1=tickers[i]
                else:
                    signal2=tickers[i - (len(result) // 2)]
                    break

        pairs.append((signal1, signal2))

    return pairs

def dataframe_interval(start, end,data):

    start_date = datetime(*start).strftime("%Y-%m-%d")
    end_date = datetime(*end).strftime("%Y-%m-%d")

    mask = (data.index > start_date) & (data.index <= end_date)

    return data.loc[mask]

def compute_pca(n_components, df, svd_solver='auto', random_state=0):
    """
    This function applies Principal Component Analysis to the df given as
    parameter

    :param n_components: number of principal components
    :param df: dataframe containing time series for analysis
    :param svd_solver: solver for PCA: see PCA documentation
    :return: reduced normalized and transposed df
    """

    if not isinstance(n_components, str):
        if n_components > df.shape[1]:
            print("ERROR: number of components larger than samples...")
            exit()

    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)

    df2 = pd.DataFrame(pca.fit_transform(df), index=df.index)
    

    return df2

def stock_screener(filename,target,sector,start,end):

    file=filename+target+'_screener.csv'
    df=pd.read_csv(file,encoding='latin1')


    mask=df['Sector'].str.contains(sector)
    mask=mask.where(pd.notnull(mask), False).tolist()
    tickers=df['Symbol']
    tickers=tickers[mask].tolist()

    file = open(filename+target+'_'+sector+'.csv', 'w+')
    file = csv.writer(file)
    tickers.insert(0,'')
    file.writerow(tickers)

    end=(2023-end)*365
    start=(2023-start)*365
    file.writerow([f'=STOCKHISTORY(B1,NOW()-{start},NOW()-{end},,0)','',f'=STOCKHISTORY(C1,NOW()-{start},NOW()-{end},,0,1)'])

    return tickers

# stock_screener('data/','s&p500','Industrials',2010,2020)

# df=pd.read_csv('data/s&p500_Financials.csv')
# df.index=df['Date']
# print(df)