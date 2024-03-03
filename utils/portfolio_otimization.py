
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from utils.utils import price_of_entire_component,dataframe_interval,coint_spread,load_args
import riskfolio as rp
import matplotlib.pyplot as plt
# setting the seed allows for reproducible results
np.random.seed(123)

import tensorflow as tf
from keras.layers import LSTM, Flatten, Dense
from keras.models import Sequential
import keras.backend as K
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco

LONG_SPREAD = 1
SHORT_SPREAD = -1
CLOSE_POSITION = 0


def plot_efficient_frontier(data):

    # Calculate daily returns from the price data
    returns = data.pct_change().dropna()

    # # Expected returns and covariance matrix
    # mu = np.array(returns.mean(), ndmin=2)
    # cov=np.cov(returns.T)
    # sigma = np.array(cov, ndmin=2)

    # # Define a range of risk aversion values
    # risk_aversion_values = np.linspace(0.001, 1000.0, num=1000)

    # # Initialize lists to store results
    # expected_returns = []
    # portfolio_std_devs = []

    # # Loop through each risk aversion value
    # for risk_aversion in risk_aversion_values:
    #     optimal_weights = markowitz_portfolio(data, obj="Utility", risk_factor=risk_aversion)

    #     # Calculate expected return and portfolio standard deviation
    #     expected_return = np.dot(mu, optimal_weights)
    #     portfolio_std_dev = np.sqrt(optimal_weights.T @ sigma @ optimal_weights)
            
    #     # Append the values to the lists
    #     expected_returns.append(expected_return*100)
    #     portfolio_std_devs.append(portfolio_std_dev*100)

    # # Convert lists to NumPy arrays and reshape them
    # expected_returns = np.squeeze(np.array(expected_returns))
    # portfolio_std_devs = np.squeeze(np.array(portfolio_std_devs))

    # # Generate random portfolios
    # num_random_portfolios = 1000000
    # random_returns = []
    # random_volatilities = []
    
    # for i in range(num_random_portfolios):
    #     np.random.seed(i)
    #     random_weights = np.random.normal(size=(len(mu.T)))  # weights sum to 1
    #     random_weights = random_weights/np.sum(random_weights)
    #     random_return = np.dot(mu, random_weights)
    #     random_volatility = np.sqrt(random_weights.T @ sigma @ random_weights)
    #     random_returns.append(random_return*100)
    #     random_volatilities.append(random_volatility*100)

    # # Plot the efficient frontier
    # plt.figure(figsize=(10, 6))

    # # Plot random portfolios (inefficient)
    # plt.scatter(random_volatilities, random_returns, color='lightgrey', alpha=0.5, s=10, label='Inefficient Portfolios')

    # # Plot efficient frontier
    # plt.plot(portfolio_std_devs, expected_returns, color='dodgerblue', linewidth=2, label='Efficient Frontier')

    # plt.xlabel('Portfolio Risk (Standard Deviation)', fontsize=12)
    # plt.ylabel('Expected Return', fontsize=12)
    # plt.title('Efficient Frontier with Inefficient Portfolios', fontsize=14)

    # plt.legend()

    # plt.grid(True)

    # # Change the style of the grid lines
    # plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    # plt.show()

def historical_returns(spread_train,  entry_l=-1, close_l=0, entry_s=1, close_s=0,**kwargs):
    # Norm spread
    spread=((spread_train-spread_train.mean())/spread_train.std()).reset_index(drop=True)

    # Get entry/exit points
    longs_entry = spread < entry_l
    longs_exit = spread > close_l
    shorts_entry = spread > entry_s
    shorts_exit = spread < close_s


    # numerical_units long/short - equivalent to the long/short_entry arrays but with integers instead of booleans
    num_units_long = pd.Series([np.nan for i in range(len(spread))])
    num_units_short = pd.Series([np.nan for i in range(len(spread))])

    num_units_long[longs_entry] = LONG_SPREAD
    num_units_long[longs_exit] = CLOSE_POSITION
    num_units_short[shorts_entry] = SHORT_SPREAD
    num_units_short[shorts_exit] = CLOSE_POSITION

    # a bit redundant, the stabilizing threshold ensures this
    num_units_long[0] = CLOSE_POSITION
    num_units_short[0] = CLOSE_POSITION

    # completes the array by propagating the last valid observation
    num_units_long = num_units_long.fillna(method='ffill')
    num_units_short = num_units_short.fillna(method='ffill')

    # concatenation of both arrays in a single decision array
    num_units = num_units_long + num_units_short
    trade_array = pd.Series(data=num_units.values)
    trade_array.iloc[-1] = CLOSE_POSITION

    return trade_array


def historical_returns3(spread_train, entry=1, close=0,**kwargs):
    # Norm spread
    spread=((spread_train-spread_train.mean())/spread_train.std()).reset_index(drop=True)

    # Get entry/exit points
    longs_entry = spread < -entry
    longs_exit = spread > -close
    shorts_entry = spread > entry
    shorts_exit = spread < close


    # numerical_units long/short - equivalent to the long/short_entry arrays but with integers instead of booleans
    num_units_long = pd.Series([np.nan for i in range(len(spread))])
    num_units_short = pd.Series([np.nan for i in range(len(spread))])

    num_units_long[longs_entry] = LONG_SPREAD
    num_units_long[longs_exit] = CLOSE_POSITION
    num_units_short[shorts_entry] = SHORT_SPREAD
    num_units_short[shorts_exit] = CLOSE_POSITION

    # a bit redundant, the stabilizing threshold ensures this
    num_units_long[0] = CLOSE_POSITION
    num_units_short[0] = CLOSE_POSITION

    # completes the array by propagating the last valid observation
    num_units_long = num_units_long.fillna(method='ffill')
    num_units_short = num_units_short.fillna(method='ffill')

    # concatenation of both arrays in a single decision array
    num_units = num_units_long + num_units_short
    trade_array = pd.Series(data=num_units.values)
    trade_array.iloc[-1] = CLOSE_POSITION

    return trade_array

def historical_returns2(spread_train,  entry_l=-1, close_l=0, entry_s=1, close_s=0, look_back=126,**kwargs):

    spread=spread_train.to_numpy()
     # Calculate the Bollinger Bands
    spread_mean = spread_train.rolling(window=look_back).mean().to_numpy()
    spread_std = spread_train.rolling(window=look_back).std().to_numpy()

    upper_band = spread_mean + (spread_std * entry_s)
    lower_band = spread_mean + (spread_std * entry_l)

    # Get entry/exit points
    longs_entry = spread < lower_band
    longs_exit = spread > spread_mean
    shorts_entry = spread > upper_band
    shorts_exit = spread < spread_mean


    # numerical_units long/short - equivalent to the long/short_entry arrays but with integers instead of booleans
    num_units_long = pd.Series([np.nan for i in range(len(spread))])
    num_units_short = pd.Series([np.nan for i in range(len(spread))])

    num_units_long[longs_entry] = LONG_SPREAD
    num_units_long[longs_exit] = CLOSE_POSITION
    num_units_short[shorts_entry] = SHORT_SPREAD
    num_units_short[shorts_exit] = CLOSE_POSITION

    # a bit redundant, the stabilizing threshold ensures this
    num_units_long[0] = CLOSE_POSITION
    num_units_short[0] = CLOSE_POSITION

    # completes the array by propagating the last valid observation
    num_units_long = num_units_long.fillna(method='ffill')
    num_units_short = num_units_short.fillna(method='ffill')

    # concatenation of both arrays in a single decision array
    num_units = num_units_long + num_units_short
    trade_array = pd.Series(data=num_units.values)
    trade_array.iloc[-1] = CLOSE_POSITION

    return trade_array

def get_portfolio_spreads(pairs,tickers,train_start,train_end,data):
    spreads = pd.DataFrame()
    df_c1 = pd.DataFrame()
    df_c2 = pd.DataFrame()

    for id,(component1, component2) in enumerate(pairs):  # get components for each pair
        # convert component1 and component2 to strings
        component1_str = '_'.join(component1)
        component2_str = '_'.join(component2)
        # Extract tickers in each component
        component1 = [(ticker in component1) for ticker in tickers]
        component2 = [(ticker in component2) for ticker in tickers]

        # Get one series for each component
        c1 = price_of_entire_component(data, component1)
        c2 = price_of_entire_component(data, component2)

        # Get series between train/test/full date intervals
        c1_train = dataframe_interval(
            train_start, train_end, c1)
        c2_train = dataframe_interval(
            train_start, train_end, c2)
        
        # Get beta coefficient and spread for train/test/full
        beta, spread_train = coint_spread(c1_train, c2_train)
        
        df_c1[f'{component1_str}_{component2_str}']=c1_train
        df_c2[f'{component1_str}_{component2_str}']=c2_train
        # add spread to DataFrame
        spreads[f'{component1_str}_{component2_str}'] = spread_train
    return spreads,df_c1,df_c2

def uniform_portfolio(N):

    return [1/N]*N

def markowitz_portfolio(data,risk_factor=0.5):
    
    # Calculate daily returns from the price data
    returns = data.pct_change().dropna()

    # Expected returns and covariance matrix
    mu = np.array(returns.mean(), ndmin=2)
    cov=np.cov(returns.T)
    sigma = np.array(cov, ndmin=2)

    # Define optimization variables
    w = cp.Variable((mu.shape[1], 1))
    std = cp.Variable(nonneg=True)
    S = sqrtm(sigma)

    constraints = [cp.sum(w) == 1]
    constraints += [cp.SOC(std, S.T @ w)]
    constraints += [w <=1, w >= 0]
    
    # Here we define risk as variance (square of standard deviation)
    risk = std**2
    ret = mu @ w

    if risk_factor == -1:
        objective = cp.Minimize(risk)
    elif risk_factor == -2:
        objective = cp.Maximize(ret/std)
    else:
        objective = cp.Maximize(ret - risk_factor * risk)


    # Create the optimization problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Optimal portfolio weights
    optimal_weights = w.value

    return optimal_weights


def riskportfolio(df_spreads,window):

    ret= df_spreads.pct_change()[window:].dropna()
    # Building the portfolio object
    port = rp.Portfolio(returns=ret)
# Get indices of NaN and inf values
    nan_indices = np.where(pd.isna(ret))
    inf_indices = np.where(np.isinf(ret))


    print("\nIndices of NaN values:")
    print(list(zip(*nan_indices)))

    print("\nIndices of inf values:")
    print(list(zip(*inf_indices)))
    # Calculating optimal portfolio

    # Risk Measures available:
    #
    # 'MV': Standard Deviation.
    # 'MAD': Mean Absolute Deviation.
    # 'MSV': Semi Standard Deviation.
    # 'FLPM': First Lower Partial Moment (Omega Ratio).
    # 'SLPM': Second Lower Partial Moment (Sortino Ratio).
    # 'CVaR': Conditional Value at Risk.
    # 'EVaR': Entropic Value at Risk.
    # 'WR': Worst Realization (Minimax)
    # 'MDD': Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio).
    # 'ADD': Average Drawdown of uncompounded cumulative returns.
    # 'CDaR': Conditional Drawdown at Risk of uncompounded cumulative returns.
    # 'EDaR': Entropic Drawdown at Risk of uncompounded cumulative returns.
    # 'UCI': Ulcer Index of uncompounded cumulative returns.
    rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM', 'CVaR',
        'EVaR', 'WR', 'MDD', 'ADD', 'CDaR', 'UCI', 'EDaR']

    method_mu='hist' # Method to estimate expected returns based on historical data.
    method_cov='hist' # Method to estimate covariance matrix based on historical data.

    # First we need to set a solver that support Mixed Integer Programming

    # Then we need to set the cardinality constraint (maximum number of assets)
    
    port.assets_stats(method_mu=method_mu, method_cov=method_cov,d=0.94)

    # Estimate optimal portfolio:

    model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
    rm = 'MV' # Risk measure used, this time will be variance
    obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe

    rf=0


    w = port.optimization(model=model, rm=rm, obj=obj,rf=rf)
    # w = port.rp_optimization(model=model)

    # port.efficient_frontier(model=model, rm=rm)

    return w.to_numpy()
def pyportfolio(df_spreads):
    pass




def plot_spread_and_returns(spreads: pd.DataFrame, threshold: float, expected_returns: pd.DataFrame):
    # Compute the mean of the first column of the spreads DataFrame
    mean = spreads.mean()
    
    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    # Plot the spread, mean, and thresholds on the first subplot
    ax1.plot(spreads, label='Spread')
    ax1.axhline(mean, color='gray', linestyle='--', label='Mean')
    ax1.axhline(mean + threshold, color='red', linestyle='--', label=f'Mean + {threshold}')
    ax1.axhline(mean - threshold, color='red', linestyle='--', label=f'Mean - {threshold}')
    ax1.legend()
    
    # Compute the cumulative expected returns
    cum_expected_returns = expected_returns.cumsum()
    
    # Plot the cumulative expected returns on the second subplot
    ax2.plot(cum_expected_returns, label='Cumulative Expected Returns')
    ax2.legend()
    
    # Set the xticks to not show the index
    ax2.set_xticklabels([])
    
    # Show the plot
    plt.show()


def clean_weights(weights,th=0.0001,uni=False):
    # Substitute all values in the array lower than the threshold for 0

    subs=1 if uni else weights

    arr = np.where(weights < th, 0, subs)
    arr_sum=np.sum(arr)
    
    
    return np.ravel(arr/arr_sum)

from statsmodels.tsa.arima.model import ARIMA
def mra(rets,spreads):
    # Substitute all values in the array lower than the threshold for 0
    pass
    # subs=1 if uni else weights

    # arr = np.where(weights < th, 0, subs)
    # arr_sum=np.sum(arr)
    
    
    # return np.ravel(arr/arr_sum)



def deep_learning_otimization(returns):

    # price=compute_prices(spread=df_spreads,**load_args("TH"))
    portfolio=LSTM_PORTFOLIO()

    w=portfolio.get_allocations(returns)

    return w



class LSTM_PORTFOLIO:
    def __init__(self):
        self.data = None
        self.model = None
        
    def __build_model(self, input_shape, outputs):
        '''
        Builds and returns the Deep Neural Network that will compute the allocation ratios
        that optimize the Sharpe Ratio of the portfolio
        
        inputs: input_shape - tuple of the input shape, outputs - the number of assets
        returns: a Deep Neural Network model
        '''
        model = Sequential([
            LSTM(64, input_shape=input_shape),
            Flatten(),
            Dense(outputs, activation='softmax')
        ])

        def sharpe_loss(_, y_pred):
            # make all time-series start at 1
            data = tf.divide(self.data, self.data[0])  
            
            # value of the portfolio after allocations applied
            portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1) 
            
            portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula

            sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)
            
            # since we want to maximize Sharpe, while gradient descent minimizes the loss, 
            #   we can negate Sharpe (the min of a negated function is its max)
            return -sharpe
        
        def utility_loss(_, y_pred):
            # make all time-series start at 1
            data = tf.divide(self.data, self.data[0])  
            
            # value of the portfolio after allocations applied
            portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1) 
            
            portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula

            # Define your risk aversion factor
            risk_aversion_factor = 0.5

            utility = K.mean(portfolio_returns) - risk_aversion_factor * K.var(portfolio_returns)
            
            # since we want to maximize Utility, while gradient descent minimizes the loss, 
            #   we can negate Utility (the min of a negated function is its max)
            return -utility
        model.compile(loss=sharpe_loss, optimizer='adam')
        return model
    
    def get_allocations(self, data: pd.DataFrame):
        '''
        Computes and returns the allocation ratios that optimize the Sharpe over the given data
        
        input: data - DataFrame of historical closing prices of various assets
        
        return: the allocations ratios for each of the given assets
        '''
        
        # data with returns
        data_w_ret = np.concatenate([ data.values[1:], data.pct_change().values[1:] ], axis=1)
        
        data = data.iloc[1:]
        self.data = tf.cast(tf.constant(data), float)
        
        if self.model is None:
            self.model = self.__build_model(data_w_ret.shape, len(data.columns))
        
        fit_predict_data = data_w_ret[np.newaxis,:]        
        self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=100, shuffle=False)
        return self.model.predict(fit_predict_data)[0]
    

def plot_portfolio(weights,pairs):
    import seaborn as sns

    # Use a list comprehension to apply the code to every pair
    labels = ['_'.join(pair[0]) + '_' + '_'.join(pair[1]) for pair in pairs]

    # Filter out labels and weights with a value of 0
    labels = [label for label, weight in zip(labels, weights) if weight != 0]
    weights = [weight for weight in weights if weight != 0]

    
    # Create a pie chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette('bright')
    wedges, texts = ax.pie(weights,colors = colors)

    # Create a legend on the side with the labels
    ax.legend(wedges, labels,
            title="Pairs",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            ncol=2)

    plt.title('Portfolio Visualization')

    plt.show()