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
import yfinance as yf
import pickle
import json

import sys
import subprocess
file_screener = 'screeners/'
file_input = 'results/'
file_output = 'results/'
file_args="modules/args.json"


LONG_SPREAD = 1
SHORT_SPREAD = -1
CLOSE_POSITION = 0
NB_TRADING_DAYS = 252

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



def coint_spread2(c1_train, c2_train,c1_test=1, c2_test=1,c1_full=1, c2_full=1):
    S1 = (np.asarray(c1_train))
    S2 = (np.asarray(c2_train))
    S2_c = sm.add_constant(S2)

    results = sm.OLS(S1, S2_c).fit()

    b = results.params[1]

    spread_train = (c1_train) - b*(c2_train)
    spread_full = (c1_full)-b*(c2_full)
    spread_test = (c1_test)-b*(c2_test)
    return b, spread_train,spread_test,spread_full

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

from statsmodels.tsa.vector_ar.vecm import coint_johansen,VECM


import math

def compute_zscore(full_spread, test_spread,interval):

    zs_interval= (len(full_spread)-len(test_spread))//1
    # zs_interval= (len(test_spread))//1
    # zs_interval=math.ceil(calculate_half_life(full_spread[:(len(full_spread)-len(test_spread))]))*2
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

def compute_zscore2(full_spread, test_spread):

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

import seaborn as sns
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
    plt.savefig(c1.name+".png")
    plt.show()

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
    plt.show()


def plot_components2(c1, c2):



    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(20, 10))

    s=c1-c2
    zspread = np.array((s - s.mean()) / s.std())
    # Adjust subplot parameters to add space at the bottom
    plt.subplots_adjust(bottom=0.2)
    sns.set_theme()
    sns.set_style("dark")
    # Increase the size of the text elements
    sns.set_context("paper", font_scale=2)

    # Plot z-spread
    axs.plot(zspread)
    axs.set_ylabel("Value")
    axs.axhline(y=2, color='black', linestyle='--', label='Short Threshold')
    axs.axhline(y=-2, color='black', linestyle=':', label='Long Threshold')
    axs.legend(loc='upper right')

   
    spread_test_indices = c1.index
    tick_indices = np.linspace(0, len(spread_test_indices) - 1, 5).astype(int)
    axs.set_xticks(tick_indices)
    # Slice the last 3 characters of each string in spread_test_indices
    axs.set_xticklabels([spread_test_indices[idx] for idx in tick_indices], rotation=45)
    axs.set_xlabel("Date")

    # Apply grid to both subplots
    axs.grid(True)
    # Set x-axis limit to cover the range from the first day to the last day of the data
    axs.set_xlim(0, len(c1.index) - 1)
    plt.tight_layout()
    plt.show()

def plot_forecasts(real, fc, horizon, model):

    model=['ARIMA','LightGBM']
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
    plt.savefig(real.columns[0]+".png")
    # Show the plot
    plt.show()


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




def features_ts(arr,name='y',diff=False):

    df = pd.DataFrame(arr.values, columns=[name],index=arr.index)

    if diff:
        df[name]=df[name].diff()

    
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

def mean_absolute_scaled_error(y_true, y_pred, y_train):
    """
    Calculate the Mean Absolute Scaled Error (MASE)
    :param y_true: array of true values
    :param y_pred: array of predicted values
    :param y_train: array of training values
    :return: MASE score
    """
    n = y_train.shape[0]
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

def hurst_rolling(x):
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0
def plot_forecast(Y_train_df,y_true, Y_hat_df, iteration, start,end,pair,models):
    
    # Plot predictions
    fig, ax = plt.subplots(1, 1, figsize = (20, 7))
    Y_hat_df = pd.merge(y_true,Y_hat_df, how='left', on=['ds'])
    plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')

    plot_df.plot(ax=ax, linewidth=2)

    ax.set_title('AirPassengers Forecast', fontsize=22)
    ax.set_ylabel('Monthly Passengers', fontsize=20)
    ax.set_xlabel('Timestamp [t]', fontsize=20)
    ax.legend(prop={'size': 15})
    ax.grid()


    pair_string = '_'.join([x[0] for x in pair])
    model_string='_'.join(models)
    filename = f'results\\img\\{start}_{end}_{pair_string}_{model_string}_{iteration}'
    filename = filename.replace(':', '_')
    plt.savefig(filename)

def plot_forecast2(real, forecast, horizon, trading_decisions,start,end,pair,models):
    
    # Shift the forecast array by the specified horizon
    shifted_forecast = pd.DataFrame(forecast).shift(horizon)

    # Plot the data
    plt.plot(real, label='Real')
    plt.plot(shifted_forecast, label='Forecast')

    # Add markers for trading decisions
    for i, decision in enumerate(trading_decisions):
        if decision == 1:
            plt.scatter(i, real[i], marker='^', color='green')
        elif decision == -1:
            plt.scatter(i, real[i], marker='v', color='red')
        elif decision == 0:
            plt.scatter(i, real[i], marker='o', color='blue')
    
    # Add the horizon value to the title
    plt.title(f'Horizon: {horizon}')
    plt.legend()


    pair_string = '_'.join([x[0] for x in pair])
    model_string='_'.join(models)
    filename = f'results\\img\\{start}_{end}_{pair_string}_{model_string}'
    filename = filename.replace(':', '_')
    plt.savefig(filename)
    plt.close()


def calculate_threshold(training_period,validation_period,c1_val,c2_val,val_fc):
        # Define a function to calculate the spread percentage change
    


    # Calculate the spread percentage change during the formation period
    x = pct_change = training_period.pct_change()

    # Calculate the positive and negative percentage change distributions
    f_plus = x[x > 0]
    f_minus = x[x < 0]

    # Calculate the decile-based and quintile-based thresholds
    alpha_L_candidates = [np.percentile(f_plus, 80), np.percentile(f_plus, 90)]
    alpha_S_candidates = [np.percentile(f_minus, 20), np.percentile(f_minus, 10)]

    return np.min(alpha_L_candidates),np.max(alpha_S_candidates)
    # Define a function to evaluate the performance of the trading model on the validation set
    def evaluate(alpha_S, alpha_L):

        decision_array = pd.Series([np.nan for i in range(len(validation_period))])

        decision_array.iloc[0] = decision_array.iloc[-1] = CLOSE_POSITION
        for day in range(len(validation_period)-1):

            delta=val_fc[day]-validation_period[day]

            if delta>alpha_L:
                decision_array[day]=LONG_SPREAD
            if delta<alpha_S:
                decision_array[day]=SHORT_SPREAD

        trade_spread(c1_val, c2_val, decision_array)

        return R_val

    # Find the best threshold combination
    best_R_val = -np.inf
    best_alpha_S = None
    best_alpha_L = None
    for alpha_S in alpha_S_candidates:
        for alpha_L in alpha_L_candidates:
            R_val = evaluate(alpha_S, alpha_L)
            if R_val > best_R_val:
                best_R_val = R_val
                best_alpha_S = alpha_S
                best_alpha_L = alpha_L

    return best_alpha_L,best_alpha_S


def trade_spread(c1, c2, trade_array, FIXED_VALUE=1000, commission=0.08,  market_impact=0.2, short_loan=1):

    NB_TRADING_DAYS=len(c1)
    # Close all positions in the last day of the trading period whether they have converged or not
    trade_array.iloc[-1] = CLOSE_POSITION

    # define trading costs
    fixed_costs_per_trade = (
        commission + market_impact) / 100  # remove percentage
    short_costs_per_day = FIXED_VALUE * \
        (short_loan / NB_TRADING_DAYS) / 100  # remove percentage

    # 2 positions, one for each component of the pair
    # The first position concerns the fist component, c1, and 2nd the c2
    stocks_in_hand = np.zeros(2)
    # tracks the evolution of the balance day by day
    cash_in_hand = np.zeros(len(trade_array))
    cash_in_hand[0] = FIXED_VALUE  # starting balance
    portfolio_value = np.zeros(len(trade_array))
    portfolio_value[0] = cash_in_hand[0]

    n_trades = 0  # how many trades were made?
    # how many profitable/unprofitable trades were made
    profitable_unprofitable = np.zeros(2)

    # how many days has this position been open?
    days_open = np.zeros(len(trade_array))

    for day, decision in enumerate(trade_array):
        # the first day of trading is excluded to stabilize the spread
        # and avoid  accessing positions out of range in the decision array when executing decision_array[day-1]
        if day == 0:
            continue  # skip the first day as mentioned above

        # at the beginning of the day we still have the cash we had the day before
        cash_in_hand[day] = cash_in_hand[day - 1]
        portfolio_value[day] = portfolio_value[day - 1]

        # at the beginning of the day the position hasn't been altered
        days_open[day] = days_open[day-1]

        # the state has changed and the TS is called to act
        if trade_array[day] != trade_array[day - 1]:

            n_trades += 1

            sale_value = stocks_in_hand[0] * \
                c1[day] + stocks_in_hand[1] * c2[day]
            # 2 closes, so 2*transaction costs
            cash_in_hand[day] += sale_value * (1 - 2*fixed_costs_per_trade)

            # both positions were closed
            stocks_in_hand[0] = stocks_in_hand[1] = 0

            days_open[day] = 0  # the new position was just opened

            if sale_value > 0:
                profitable_unprofitable[0] += 1  # profit
            elif sale_value < 0:
                profitable_unprofitable[1] += 1  # loss

            if decision == SHORT_SPREAD:  # if the new decision is to SHORT the spread
                # if the previous trades lost money I have less than FIXED VALUE to invest
                value_to_buy = min(FIXED_VALUE, cash_in_hand[day])
                # long c2
                cash_in_hand[day] += -value_to_buy
                stocks_in_hand[1] = value_to_buy / c2[day]
                # short c1
                cash_in_hand[day] += value_to_buy
                stocks_in_hand[0] = -value_to_buy / c1[day]
                # apply transaction costs (with 2 operations made: short + long)
                cash_in_hand[day] -= 2*value_to_buy*fixed_costs_per_trade

            elif decision == LONG_SPREAD:  # if the new decision is to LONG the spread
                value_to_buy = min(FIXED_VALUE, cash_in_hand[day])
                # long c1
                cash_in_hand[day] += -value_to_buy
                stocks_in_hand[0] = value_to_buy / c1[day]
                # short c2
                cash_in_hand[day] += value_to_buy
                stocks_in_hand[1] = -value_to_buy / c2[day]
                # apply transaction costs (with 2 operations made: short + long)
                cash_in_hand[day] -= 2 * \
                    value_to_buy * fixed_costs_per_trade

        # short rental costs are applied daily!
        # means there's an open position
        if stocks_in_hand[0] != 0 or stocks_in_hand[1] != 0:
            cash_in_hand[day] -= short_costs_per_day
            days_open[day] += 1
        # at the end of the day, the portfolio value takes in consideration the value of the stocks in hand
        portfolio_value[day] = cash_in_hand[day] + \
            stocks_in_hand[0] * c1[day] + stocks_in_hand[1] * c2[day]

    return n_trades, cash_in_hand, portfolio_value, days_open, profitable_unprofitable


def calculate_dynamic_beta(c1, c2, test_c1, test_c2):
    

    offset = c1.index.get_loc(test_c1.index[0])

    # Calculate the beta using the training data
    beta,_ = coint_spread(c1[:offset], c2[:offset])


    # Create an array to store the betas for the whole data
    all_betas = np.empty(len(c1))

    # Fill the array with the beta value from the training set
    all_betas[:offset] = beta


    # Calculate the betas for the test set
    for i in range(offset, len(c1)):
        beta,_ = coint_spread(c1[:i], c2[:i])

        all_betas[i] = beta


    spread_full=c1-all_betas*c2
    dynamic_beta=all_betas[offset:]
    spread_test=test_c1-dynamic_beta*test_c2

    return dynamic_beta,spread_full,spread_test


def trade_spread(c1_test, c2_test, trade_array, FIXED_VALUE=1000, commission=0.08,  market_impact=0.2, short_loan=1,beta=1,sizing=1,leverage=1):

    if sizing:
        beta=1

    # Close all positions in the last day of the trading period whether they have converged or not
    trade_array.iloc[-1] = CLOSE_POSITION

    # define trading costs
    fixed_costs_per_trade = (
        commission + market_impact) / 100  # remove percentage
    short_costs_per_day = FIXED_VALUE * \
        (short_loan / 252) / 100  # remove percentage

    # 2 positions, one for each component of the pair
    # The first position concerns the fist component, c1, and 2nd the c2
    stocks_in_hand = np.zeros(2)
    # tracks the evolution of the balance day by day
    cash_in_hand = np.zeros(len(trade_array))
    cash_in_hand[0] = FIXED_VALUE  # starting balance
    portfolio_value = np.zeros(len(trade_array))
    portfolio_value[0] = cash_in_hand[0]

    n_trades = 0  # how many trades were made?
    # how many profitable/unprofitable trades were made
    profitable_unprofitable = np.zeros(2)

    # how many days has this position been open?
    days_open = np.zeros(len(trade_array))

    for day, decision in enumerate(trade_array):
        # the first day of trading is excluded to stabilize the spread
        # and avoid  accessing positions out of range in the decision array when executing decision_array[day-1]
        if day == 0:
            continue  # skip the first day as mentioned above

        # at the beginning of the day we still have the cash we had the day before
        cash_in_hand[day] = cash_in_hand[day - 1]
        portfolio_value[day] = portfolio_value[day - 1]

        # at the beginning of the day the position hasn't been altered
        days_open[day] = days_open[day-1]

        # the state has changed and the TS is called to act
        if trade_array[day] != trade_array[day - 1]:

            n_trades += 1

            sale_value = stocks_in_hand[0] * c1_test[day]+stocks_in_hand[1] * c2_test[day]
            # apply transaction costs
            cash_in_hand[day] += sale_value * (1 - fixed_costs_per_trade)

            # both positions were closed
            stocks_in_hand[0] = stocks_in_hand[1] = 0

            days_open[day] = 0  # the new position was just opened

            if sale_value > 0:
                profitable_unprofitable[0] += 1  # profit
            elif sale_value < 0:
                profitable_unprofitable[1] += 1  # loss

            if decision == SHORT_SPREAD:  # if the new decision is to SHORT the spread
                # if the previous trades lost money I have less than FIXED VALUE to invest
                value_to_buy = min(FIXED_VALUE,max(cash_in_hand[day],0)) * leverage
                # long c2
                cash_in_hand[day] += -value_to_buy * beta / (beta+1)
                stocks_in_hand[1] = value_to_buy * beta / (beta+1) / c2_test[day] 
                # short c1
                cash_in_hand[day] += value_to_buy / (beta+1)
                stocks_in_hand[0] = -value_to_buy / (beta+1) / c1_test[day]
                # apply transaction costs (with 2 operations made: short + long)
                cash_in_hand[day] -= value_to_buy*fixed_costs_per_trade

            elif decision == LONG_SPREAD:  # if the new decision is to LONG the spread
                value_to_buy = min(FIXED_VALUE, max(cash_in_hand[day],0)) * leverage
                # long c1
                cash_in_hand[day] += -value_to_buy / (beta+1)
                stocks_in_hand[0] = value_to_buy / (beta+1) / c1_test[day] 
                # short c2
                cash_in_hand[day] += value_to_buy * beta / (beta+1)
                stocks_in_hand[1] = -value_to_buy * beta / (beta+1) / c2_test[day]  
                # apply transaction costs (with 2 operations made: short + long)
                cash_in_hand[day] -= value_to_buy*fixed_costs_per_trade

        # short rental costs are applied daily!
        # means there's an open position
        if stocks_in_hand[0] != 0 or stocks_in_hand[1] != 0:
            cash_in_hand[day] -= short_costs_per_day
            days_open[day] += 1
        # at the end of the day, the portfolio value takes in consideration the value of the stocks in hand
        portfolio_value[day] = cash_in_hand[day] + \
            stocks_in_hand[0] * c1_test[day] + stocks_in_hand[1] * c2_test[day]

    return n_trades, cash_in_hand, portfolio_value, days_open, profitable_unprofitable


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
            "callback": 100,
            "seed": 8,
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
            "batch_error": 25,
            "model": [
                "light3"
            ],
            "horizon": 1
        },
        "TRADING": {
            "commission": 0.05,
            "market_impact": 0,
            "short_loan": 0,
            "leverage": 2,
            "weights": "ga"
        }
    }
    for alg, arg_dict in default.items():
        for arg, value in arg_dict.items():
            change_args(alg, arg, value)