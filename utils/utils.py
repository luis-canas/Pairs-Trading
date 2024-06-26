import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import statsmodels.api as sm
import random
from os import makedirs
from os.path import isfile, exists
from scipy.stats import skew,zscore, boxcox,kurtosis as kur
from scipy.stats.mstats import winsorize
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import yfinance as yf
import pickle
import json
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
import sys
import subprocess
import seaborn as sns
import math
file_screener = 'screeners/'
file_input = 'results/'
file_output = 'results/'
file_args="modules/args.json"
file_image = 'results/'

LONG_SPREAD = 1
SHORT_SPREAD = -1
CLOSE_POSITION = 0
NB_TRADING_DAYS = 252

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Ignore ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
# Ignore specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

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



def coint_spread(c1_train, c2_train,c1_test=1, c2_test=1,c1_full=1, c2_full=1):
    S1 = np.log(np.asarray(c1_train))
    S2 = np.log(np.asarray(c2_train))
    S2_c = sm.add_constant(S2)

    results = sm.OLS(S1, S2_c).fit()

    b = results.params[1]

    spread_train = np.log(c1_train) - b*np.log(c2_train)
    spread_full = np.log(c1_full)-b*np.log(c2_full)
    spread_test = np.log(c1_test)-b*np.log(c2_test)
    return b, spread_train,spread_test,spread_full



def compute_zscore(full_spread, test_spread,interval):

    zs_interval=interval

    i = test_spread.index[0]
    offset = full_spread.index.get_loc(i)

    norm_spread = np.zeros(len(test_spread))

    mean = np.zeros(len(test_spread))
    std = np.zeros(len(test_spread))
    t_spread = np.zeros(len(test_spread))

    for day, daily_value in enumerate(test_spread):
        
        spread_to_consider = full_spread[(day+1) + offset - zs_interval: (day+1) + offset]

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


def get_data(index):
    compute_index=False

    file=file_screener+f'{index}_screener.csv'
    isExist = exists(file)
    if isExist:
        # Load the data from a pickle file
        screener = pd.read_csv(file, encoding='latin1')

    else:
        sys.exit("Screener not found!")

    file=file_screener+'data.pkl'
    isExist = exists(file)
    if isExist:
        # Load the data from a pickle file
        with open(file, 'rb') as f:
            data = pickle.load(f)
    else:
        data={}

    if compute_index or index+'_series' not in data or index+'_sector' not in data:

        unique = pd.unique(screener['Tickers'].str.split(',', expand=True).values.ravel('K')).tolist()
        tickerSymbols = [value for value in unique if isinstance(value, str)]

        # Initialize an empty dictionary to store the sectors
        sectors = {}

        tickers=tickerSymbols.copy()

        # Loop through the list of ticker symbols
        for tickerSymbol in tickerSymbols:
            try:
                # Get the ticker data
                tickerData = yf.Ticker(tickerSymbol)

                # Get the sector
                sector = tickerData.info['sector']

                # If the sector is not in the dictionary, add it
                if sector not in sectors:
                    sectors[sector] = []

                # Add the ticker symbol to the list of tickers in the sector
                sectors[sector].append(tickerSymbol)
            except Exception as e:
                tickers.remove(tickerSymbol)

        start=screener['Date'].iloc[0]
        end=screener['Date'].iloc[-1]
        
        close = yf.download(tickers, start=start, end=end,auto_adjust=True,ignore_tz=True)['Close']
        close = close.reset_index()
        close['Date'] = close['Date'].dt.normalize()
        close = close.set_index('Date')

        data.update({index+'_series': close,index+'_sector': sectors})

        # Save the array to a pickle file
        with open(file, 'wb') as f:
            pickle.dump(data, f)

        return close,screener,sectors
    else:
        return data[index+'_series'],screener,data[index+'_sector']
    
def get_membership(data,membership,today):

    # Get the previous or equal date of today in the membership DataFrame
    previous_date = membership[membership['Date'] <= today]['Date'].iloc[-1]

    # Get the tickers inside of it to a list
    tickers = membership[membership['Date'] == previous_date]['Tickers'].str.split(',').sum()

    # Apply a mask of those values to the price series with all the assets
    df_price_filtered = data.filter(items=tickers)

    return df_price_filtered

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


def plot_positions(spread_test, spread_full, positions, window, profit_loss, portfolio_value,c1,c2):
    profit_loss = np.array(profit_loss)
    zspread = compute_zscore(spread_full, spread_test, window)[0]
    i = spread_test.index[0]
    offset = spread_full.index.get_loc(i)

    label_indices = np.linspace(0, len(zspread)-1, 10).astype(int)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(16, 8))

    # s=c1-c2
    # zspread = np.array((s - s.mean()) / s.std())
    # Adjust subplot parameters to add space at the bottom
    plt.subplots_adjust(bottom=0.2)
    sns.set_theme()
    sns.set_style("white")
    # Increase the size of the text elements
    sns.set_context("paper", font_scale=2)

    # Plot z-spread
    axs[0].plot(zspread, label='Normalized Spread')
    axs[0].set_ylabel("Value")
    axs[0].axhline(y=2, color='black', linestyle='--', label='Short Threshold')
    axs[0].axhline(y=-2, color='black', linestyle=':', label='Long Threshold')
    axs[0].set_xlim(positions.index.min(), positions.index.max())
    axs[0].legend()

    # Plot position
    axs[1].step(range(len(positions)), positions, label='Position', where='post')
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Position")
    axs[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Set y-axis ticks to be indices of spread_test
    spread_test_indices = spread_test.index
    tick_indices = np.linspace(0, len(spread_test_indices) - 1, 12).astype(int)

    # Exclude the first and last points
    tick_indices = tick_indices[1:-1]
    axs[1].set_xticks(tick_indices)
    # Slice the last 3 characters of each string in spread_test_indices
    axs[1].set_xticklabels([spread_test_indices[idx] for idx in tick_indices], rotation=30)
    # Apply grid to both subplots
    axs[0].grid(True)
    axs[1].grid(True)

    # Initialize variables for iterating through profit_loss
    current_position = 0

    # Iterate through profit_loss to determine color for each step
    for i, color in enumerate(profit_loss):
        while current_position < len(positions) and positions[current_position] == 0:
            current_position += 1

        start_index = current_position

        while current_position < len(positions)-1 and positions[current_position] == positions[current_position+1]:
            current_position += 1

        current_position += 1

        end_index = current_position
        # Plot step
        axs[1].axvspan(start_index, min(end_index, len(positions) - 1), facecolor='green' if color == 1 else 'red', alpha=0.3)

    # Legend for the second graph
    legend_elements = [plt.Rectangle((0,0),1,1,facecolor='green',alpha=0.3,label='Profit'),
                       plt.Rectangle((0,0),1,1,facecolor='red',alpha=0.3,label='Loss')] + axs[1].get_legend_handles_labels()[0]
    axs[1].legend(handles=legend_elements, loc='upper left')

    # Adjust legend size
    axs[0].legend(handles=axs[0].get_legend_handles_labels()[0], loc='upper left', prop={'size': 12})
    axs[1].legend(handles=legend_elements, loc='upper left', prop={'size': 12})
    # Set aspect ratio of the axes

    axs[0].set_aspect('auto')
    axs[1].set_aspect('auto')
    plt.tight_layout()
    plt.savefig(file_image+c1.name+"_positions"+".png")
    plt.close()

def plot_components(c1, c2):
    # Separate the names
    name1, name2 = c1.name.split("_")
    plt.figure()
    # Plot z-spread for c1
    plt.plot(c1, label=name1)
    
    # Plot z-spread for c2
    plt.plot(c2, label=name2)
    
    # Set x and y labels
    plt.xlabel("Year")
    plt.ylabel("Price ($)")
    
    # Extract years from index
    years = sorted(set([date[:4] for date in c1.index]))
    
    # Add an additional label for the next year after the last index
    last_index = len(c1.index) - 1
    last_year = int(c1.index[last_index][:4])
    next_year_label = str(last_year + 1)
    years.append(next_year_label)
    
    # Calculate the number of points per year
    points_per_year = 250
    
    # Set x-axis ticks and labels
    tick_locations = [i * points_per_year for i in range(len(years))]
    plt.xticks(tick_locations, years, rotation=45)
    
    # Set x-axis limit to cover the range from the first day to the last day of the data
    plt.xlim(0, len(c1.index) - 1)
    
    # Add legend
    plt.legend()

    # Set plot theme and style
    sns.set_theme()
    sns.set_style("white")
    
    # Increase the size of the text elements
    sns.set_context("paper", font_scale=2)
    
    # Show grid
    plt.grid(True)
    plt.tight_layout()

    # Show plot
    plt.savefig(file_image+c1.name+"_components"+".png")
    plt.close()



def plot_forecasts(real, fc, horizon, model):

    import seaborn as sns
    import matplotlib.ticker as ticker

    # Calculate the mean of the real values
    y_true = np.mean(real.values, axis=1)

    # Define the positions on the x-axis
    index = range(0, len(y_true[:-horizon]))

    plt.figure(figsize=(16, 8))
    plt.plot(index, y_true[:-horizon], label='Real', color='blue')

    # Define a list of colors for the forecast models
    num_forecasts = len(fc)
    forecast_colors = sns.color_palette("husl", num_forecasts)

    for i, (forecast, mod) in enumerate(zip(fc, model)):
        fc_mean = np.mean(forecast, axis=1)
        plt.plot(index, fc_mean[:-horizon], label=mod, color=forecast_colors[i])

    # Add labels, legend, and grid
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.subplots_adjust(bottom=0.2)
    plt.grid(True)

    # Set plot theme and style
    sns.set_theme()
    sns.set_style("white")
    sns.set_context("paper", font_scale=2)

    # Set x-axis ticks and labels
    real_indices = real.index
    tick_indices = np.linspace(0, len(real) - 1, 12).astype(int)

    # Exclude the first and last points
    tick_indices = tick_indices[1:-1]
    plt.xticks(tick_indices, [real_indices[idx] for idx in tick_indices], rotation=30)

    # Set x-axis limit to cover the range from the first day to the last day of the data
    plt.xlim(0, len(real.index) - 1-horizon)
    plt.tight_layout()
    plt.savefig(file_image+real.columns[0]+"_forecasts"+".png")
    plt.close()


def max_accumulate(arr):
    max_val = arr[0]
    max_cumulative = np.empty_like(arr)
    for i in range(arr.shape[0]):
        max_val = max(max_val, arr[i])
        max_cumulative[i] = max_val
    return max_cumulative


def max_drawdown(s):

    try:
        i = np.argmax(max_accumulate(s) - s) # end of the period
        j = np.argmax(s[:i]) # start of period

        mdd = ((s[j] - s[i])/s[j] ) * 100

    except:
        mdd=100
        i=j=0

    return mdd, i, j


def annualized_stats(cash,date):

    rf = {
        1987: 5.775 / 100,
        1988: 6.6675 / 100,
        1989: 8.111666667 / 100,
        1990: 7.493333333 / 100,
        1991: 5.375 / 100,
        1992: 3.431666667 / 100,
        1993: 2.9975 / 100,
        1994: 4.246666667 / 100,
        1995: 5.49 / 100,
        1996: 5.005833333 / 100,
        1997: 5.060833333 / 100,
        1998: 4.776666667 / 100,
        1999: 4.638333333 / 100,
        2000: 5.816666667 / 100,
        2001: 3.388333333 / 100,
        2002: 1.6025 / 100,
        2003: 1.010833333 / 100,
        2004: 1.371666667 / 100,
        2005: 3.146666667 / 100,
        2006: 4.726666667 / 100,
        2007: 4.353333333 / 100,
        2008: 1.365 /100 ,
        2009: 0.15 /100 ,
        2010: 0.136666667 /100 ,
        2011: 0.0525 /100 ,
        2012: 0.085833333 /100 ,
        2013: 0.058333333 /100 ,
        2014: 0.0325 /100 ,
        2015: 0.0525 /100 ,
        2016: 0.3175 /100 ,
        2017: 0.930833333 /100 ,
        2018: 1.939166667 /100 ,
        2019: 2.060833333 /100 ,
        2020: 0.365 /100 ,
        2021: 0.044166667 /100 ,
        2022: 2.021666667 /100 
    }


    date_key = datetime.strptime(date, '%Y-%m-%d').year

    # Compute returns
    ret = pd.Series(np.diff(cash) / cash[:-1], index=pd.date_range(start=date, periods=len(cash)-1))

    # Resample returns to daily frequency
    daily_index = ret.resample('D').sum().dropna().index
    daily_ret = (ret + 1).resample('D').prod() - 1

    # Remove added days from resample
    daily_ret = daily_ret.loc[daily_index]
    # Number of trading days in a year
    n_days = NB_TRADING_DAYS


    # Calculate annualized return as the cumulative return of the last period
    annualized_ret = (np.cumprod(1 + ret) - 1)[-1]
    annualized_vol = (np.std(daily_ret) * np.sqrt(n_days))

    if annualized_vol != 0:
        try:
            risk_free_rate = rf[date_key]
            sharpe_ratio = (annualized_ret - risk_free_rate) / annualized_vol
        except:
            sharpe_ratio = annualized_ret / annualized_vol
    else:
        sharpe_ratio=0
    
    mdd=max_drawdown(cash)[0]
    kurtosis,skewness,median,mean,max,min=return_stats(daily_ret)

    return daily_ret.values,sharpe_ratio,annualized_ret*100,mdd,annualized_vol,kurtosis,skewness,median,mean,max,min

def return_stats(returns):

    zs=zscore(returns)
    bound=11
    mask = (zs >= -bound) & (zs <= bound)
    # Find the highest and lowest values

    kurtosis=kur(returns[mask])
    skewness=skew(returns[mask])
 



    return kurtosis,skewness,np.median(returns) * 100, np.mean(returns) * 100 , np.max(returns) * 100 , np.min(returns)* 100




def features_ts(arr,name='y'):

    df = pd.DataFrame(arr.values, columns=[name],index=arr.index)

    
    delta = df[name].diff()
    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.abs().rolling(14).mean()
    rs = avg_gain / avg_loss
    # df['RSI'] = 100 - (100 / (1 + rs))
        
    exp1 = df[name].ewm(span=12, adjust=False).mean()
    exp2 = df[name].ewm(span=26, adjust=False).mean()
    
    n = 10
   
    df['daily_return'] = df[name].diff()
   

    df['dEMA_9'] = df[name].ewm(span=9).mean()
    df['dEMA_21'] = df[name].ewm(span=21).mean()
    df['dSMA_5'] = df[name].rolling(window=10).mean()
    df['dSMA_10'] = df[name].rolling(window=20).mean()
    df['dSMA_15'] = df[name].rolling(window=40).mean()
    sma = df[name].rolling(window=20).mean()
    std = df[name].rolling(window=20).std()
   
    delta = df[name].diff()
    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.abs().rolling(14).mean()
    rs = avg_gain / avg_loss
    exp1 = df[name].ewm(span=12, adjust=False).mean()
    exp2 = df[name].ewm(span=26, adjust=False).mean()
    macd = exp1-exp2
 
    n = 10
   
    close = df[name]
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df['RSI'] = rsi.fillna(0)
    EMA_12 = pd.Series(df[name].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(df[name].ewm(span=26, min_periods=26).mean())
    df['MACD'] = pd.Series(EMA_12 - EMA_26)
    df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())
    return df



def normal(array):
    return array/np.sum(array)


def run(pairs_alg,trading_alg,index,sector,start_date,end_date,months_trading,months_forming):

    command = ["python", "main.py",
           "--pairs_alg", pairs_alg,
           "--trading_alg", trading_alg,
           "--index", index,
           "--sector", sector,
           "--start_date", f"{start_date[0]},{start_date[1]},{start_date[2]}",
           "--end_date", f"{end_date[0]},{end_date[1]},{end_date[2]}",
           "--months_trading", str(months_trading),
           "--months_forming", str(months_forming)]
    
    execute = subprocess.run(command, capture_output=True, text=True)

    if execute.returncode != 0:
        print("Error occurred.")
        print(execute.stderr)
    else:
        print("Completed successfully.")
        print(execute.stdout)


def default_args():

    default = {
        "DIST": {
            "pair_number": 10
        },
        "COINT": {
            "pvalue_threshold": 0.01,
            "hurst_threshold": 0.5
        },
        "GA": {
            "gen": 1000,
            "pop": 100,
            "cv": 0.7,
            "mt": 0.1,
            "percentage": 0.1
        },
        "TH": {
            "entry_l": -2,
            "entry_s": 2,
            "close_l": 0,
            "close_s": 0,
            "window": 21
        },
        "FA": {
            "batch": 1,
            "model": [
                "lightgbm"
            ],
            "horizon": 5
        },
        "TRADING": {
            "transaction_cost": 0.05,
            "weights": "ga"
        }
    }
    
    for alg, arg_dict in default.items():
        for arg, value in arg_dict.items():
            change_args(alg, arg, value)


def portfolio_plots(opt,port,returns,obj,pop,gen,sector,window,percentage,algorithm,optimization):

    fs=25

    plt.subplots(figsize=(15, 10))
    sns.set_theme()
    sns.set_style("white")
    sns.set_context("paper", font_scale=5)
    plt.plot(np.arange(len(opt)), opt, linestyle='--',color='black')

    # Increase font size for the x-axis label
    plt.xlabel('Generation', fontsize=fs)

    # Increase font size for the y-axis label
    plt.ylabel('Fitness', fontsize=fs)

    # Increase font size for the tick labels on both axes
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.tight_layout()
    plt.savefig(file_image+"convergence"+returns.index[0]+'_'+returns.index[-1]+".png")
    plt.close()

    rets = returns.pct_change()[window:].dropna()
    cov=np.cov(rets.T)
    sigma = np.array(cov, ndmin=2)
    mu=np.array(rets.mean(), ndmin=2)

    ro=len(port)
    co=2
    ret_risk=np.zeros(shape=(ro,co))
    for n in range(ro):

        x,asset=np.reshape(port[n],newshape=(2,-1))
        asset=np.around(asset).astype(int)
        weights=np.zeros((rets.shape[1],1))
        weights[asset]=x.reshape(-1,1)
        weights= normal(weights)
    

        daily_ret=(mu @ weights)
        daily_risk=np.sqrt(weights.T @ cov @ weights)
        ret_risk[n]= ((daily_ret + 1) ** NB_TRADING_DAYS - 1)*100, (daily_risk * np.sqrt(NB_TRADING_DAYS))*100

    fs=25
    sns.set_theme()
    sns.set_style("white")
    sns.set_context("paper", font_scale=2)
    # Plot return versus risk
    fig, ax = plt.subplots(figsize=(16, 6))
                # Set the seaborn theme

    # Define a custom colormap with a smooth gradient from purple to blue to green to yellow
    colors = ['#800080', '#0000FF', '#00FF00', '#FFFF00']  # Purple to blue to green to yellow gradient
    cmap_name = 'custom_gradient'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors)
    # Create a gradient array based on the position in the array
    gradient = np.linspace(0, len(ret_risk[:, 1]),len(ret_risk[:, 1]))
    # Plot the first set of scatter points with the custom gradient
    sc = ax.scatter(ret_risk[:, 1], ret_risk[:, 0], marker='o', c=gradient, alpha=0.6, cmap=cm,s=25)
    # Add a label to the second set of scatter points
    # Set the text position relative to the data coordinates
    text_offset = 0.1  # Adjust as needed
    text_x = ret_risk[-1, 1] + text_offset*1.6
    text_y = ret_risk[-1, 0] - text_offset 

    ax.text(text_x, text_y, 'Optimal Portfolio', ha='right', va='top', color='black')
    # Customize plot labels and title
    ax.set_xlabel('Volatility (%)')
    ax.set_ylabel('Annual Return (%)')
    # ax.grid(True)
    # Create a ScalarMappable object for the custom gradient
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])  # An empty array is needed
    # Create a color bar for the custom gradient
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Generation')
    # Create a custom formatter for the color bar ticks to multiply by gen
    formatter = FuncFormatter(lambda x, _: '{:.0f}'.format(x * gen))
    cbar.ax.yaxis.set_major_formatter(formatter)


    # Set the font size of the color bar label
    cbar.ax.set_ylabel('Generation')
    # Increase font size for the tick labels on both axes

    plt.tight_layout()
    # plt.grid(True)
    # Show the plot
    plt.savefig(file_image+"portfolio"+str(percentage*100)+returns.index[0]+'_'+returns.index[-1]+".png")
    plt.close()


    file2='results/'+'gaexpected.pkl'

    isExist = exists(file2)

    if isExist:
        # Load the data from a pickle file
        with open(file2, 'rb') as f:
            weights_cint2 = pickle.load(f)
    else:
        weights_cint2={}


    card=[0.1,0.3,0.5]

    rr=[]
    for c in card:

        string2=returns.index[0]+'_'+returns.index[-1]+obj+str(pop)+str(gen)+sector+str(c)

        if (string2 not in weights_cint2):
            cardinality=math.ceil(returns.shape[1]*c)
            

            # Optimize patterns
            results = minimize(optimization, algorithm, get_termination("n_eval", pop*gen),seed=8, save_history=True, verbose=False)

            n_evals = np.array([e.evaluator.n_eval for e in results.history]).reshape(-1,1)

            fitness=[]
            portfolio = []
            for e in results.history:
                w = [ind.X for ind in e.pop]
                portfolio.append(w)

                f = [ind.F for ind in e.pop]
                fitness.append(f)

            fitness=-np.squeeze(np.array(fitness))
            portfolio=(np.array(portfolio))

            best_individual_indices = np.argmax(fitness, axis=1)

            mask = np.zeros_like(fitness, dtype=bool)
            # Set True for the best column index for each row
            mask[np.arange(len(best_individual_indices)), best_individual_indices] = True

            opt = fitness[mask]
            port= portfolio[mask]


            rets = returns.pct_change()[window:].dropna()
            cov=np.cov(rets.T)
            sigma = np.array(cov, ndmin=2)
            mu=np.array(rets.mean(), ndmin=2)

            ro=len(port)
            co=2
            ret_risk=np.zeros(shape=(ro,co))
            for n in range(ro):

                x,asset=np.reshape(port[n],newshape=(2,-1))
                asset=np.around(asset).astype(int)
                weights=np.zeros((rets.shape[1],1))
                weights[asset]=x.reshape(-1,1)
                weights= normal(weights)
            

                daily_ret=(mu @ weights)
                daily_risk=np.sqrt(weights.T @ cov @ weights)
                ret_risk[n]= ((daily_ret + 1) ** NB_TRADING_DAYS - 1)*100, (daily_risk * np.sqrt(NB_TRADING_DAYS))*100

            weights_cint2.update({string2: ret_risk})
            # y to a pickle file
            with open(file2, 'wb') as f:
                pickle.dump(weights_cint2, f)

        rr.append(weights_cint2[string2])

        
              
    fs = 3
    sns.set_context("paper", font_scale=fs)

    # Set plot parameters
    h, w = 10, 8
    lt = 1

    # Loop through each plot
    for j, (data_type, ylabel, filename) in enumerate([("returns", "Expected Return (%)", "ret.png"),
                                        ("volatility", "Expected Volatility (%)", "vol.png"),
                                        ("sharpe_ratio", "Expected Sharpe Ratio", "sr.png")]):
        plt.figure(figsize=(h, w))
        sns.set_style("white")
        for i in range(len(card)):
            
            plt.plot(rr[i][:, j], label=f'k={card[i]}', linewidth=lt) if j < 2 else plt.plot(rr[i][:, 0]/rr[i][:, 1], label=f'k={card[i]}', linewidth=lt)

        plt.xlabel('Generation')  # Adjust the font size of the x-axis label
        plt.ylabel(ylabel )  # Adjust the font size of the y-axis label
        plt.tick_params(axis='both', which='major')  # Adjust the font size of tick labels
        plt.legend()
        plt.xlim(0, len(rr[0]))
        plt.tight_layout()
        plt.savefig(file_image + returns.index[0]+'_'+returns.index[-1] + filename)
        plt.close()
