

import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import compute_zscore
from utils.portfolio_otimization import historical_returns
import matplotlib.ticker as ticker
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV,train_test_split

import numpy as np
import tensorflow as tf
import math
import itertools

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,confusion_matrix


NB_TRADING_DAYS = 252


from sklearn.model_selection import GridSearchCV

import lightgbm as lgb
from utils.portfolio_otimization import historical_returns,riskportfolio
from utils.utils import trade_spread
import pickle
from os.path import isfile, exists

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Ignore ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
# Ignore specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

def LGBM(X,Y,offset,horizon,batch,batch_error):

    X_train, y_train, =X[:offset-horizon].to_numpy(),Y[:offset-horizon].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,shuffle=True)

    # parameters = {
    #   'n_estimators': [400],
    #   'learning_rate': [0.001,0.01, 0.1],
    #   'max_depth': [8],
    #   'gamma': [0.005, 0.01, 0.02],
    #   'random_state': [42]
    # }

    parameters = {
      'n_estimators': [400,800],
      'learning_rate': [0.001,0.01, 0.1],
      'max_depth': [4,8,16,32],
      'num_leaves': [2, 8, 32,256,1024],
      "reg_alpha":[0.1], 
      "reg_lambda":[1.0],
      'random_state': [42]
    }

    eval_set = [(X_train, y_train), (X_val, y_val)]
    model = lgb.LGBMRegressor(verbose=-1,reg_alpha=0.1, reg_lambda=1.0,early_stopping_rounds=10)
    clf = GridSearchCV(model, parameters)

    clf.fit(X_train, y_train, eval_set=eval_set)


    model = lgb.LGBMRegressor(**clf.best_params_, verbose=-1)
    # if True:
    #     plot_val_loss(X_train,y_train,eval_set,clf.best_params_)
    forecast=[]
    for i in range(offset, len(X), batch):
        # Predict on current batch
        end = min(i + batch, len(X))
        end_train = i-horizon+1
        X_train, X_test, y_train, y_test=X[i-(offset):end_train].to_numpy(),X[i:end].to_numpy(),Y[i-(offset):end_train].to_numpy(),Y[i:end].to_numpy()

        # split the data into training, validation, and test sets
        # X_val,  X_train,y_val ,  y_train= train_test_split(X_train, y_train, train_size=0.2, random_state=0,shuffle=True)

        # eval_set = [(X_train, y_train), (X_val, y_val)]

        model.fit(X_train, y_train)

        forecast.extend(model.predict(X_test))
        
    forecast=np.array(forecast).reshape(-1, 1)


    N=batch_error

    errors = np.empty(forecast.shape)

    errors[:N]=clf.best_score_
    y_test=Y[offset:].to_numpy()
    # calculate RMSE for each batch of N values
    for i in range(N, len(forecast), N):
        rmse = np.sqrt(np.mean((forecast[i:i+N] - y_test[i:i+N]) ** 2))
        errors[i:i+N]=rmse
    df = pd.DataFrame(errors)
    df.fillna(method='ffill', inplace=True)
    errors = df.to_numpy()
    return forecast

def light(X,Y,offset,horizon,batch,batch_error):

    X_train, y_train, =X[:offset-horizon].to_numpy(),Y[:offset-horizon].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    parameters = {
      'verbose': [-1],
      'n_estimators': [2000],
      'learning_rate': [0.001,0.01, 0.1],
      'num_leaves': [8, 32,128],
      "reg_alpha":[0, 1e-1, 1 ], 
      "reg_lambda":[0, 1e-1, 1],
      'random_state': [42]
    }

    eval_set = [(X_train, y_train), (X_val, y_val)]
    model = lgb.LGBMRegressor()
    clf = GridSearchCV(model, parameters)

    clf.fit(X_train, y_train, eval_set=eval_set,callbacks=[lgb.early_stopping(stopping_rounds=20,verbose=False)])


    model = lgb.LGBMRegressor(**clf.best_params_)
    # if True:
    #     plot_val_loss(X_train,y_train,eval_set,clf.best_params_)
    forecast=[]
    for i in range(offset, len(X), batch):
        # Predict on current batch
        end = min(i + batch, len(X))
        end_train = i-horizon+1
        X_train, X_test, y_train, y_test=X[i-(offset):end_train].to_numpy(),X[i:end].to_numpy(),Y[i-(offset):end_train].to_numpy(),Y[i:end].to_numpy()

        # split the data into training, validation, and test sets
        X_val,  X_train,y_val ,  y_train= train_test_split(X_train, y_train, train_size=0.2, random_state=0)

        eval_set = [(X_train, y_train), (X_val, y_val)]

        model.fit(X_train, y_train,eval_set=eval_set,callbacks=[lgb.early_stopping(stopping_rounds=20,verbose=False)])

        forecast.extend(model.predict(X_test))
        
    forecast=np.array(forecast).reshape(-1, 1)

    return forecast

def light3(X,Y,offset,horizon,batch,batch_error):

    X_train, y_train, =X[:offset-horizon].to_numpy(),Y[:offset-horizon].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    parameters = {
      'verbose': [-1],
      'n_estimators': [2000],
      'learning_rate': [0.001,0.01, 0.1],
      'num_leaves': [8, 32,128],
      "reg_alpha":[0, 1e-1, 1 ], 
      "reg_lambda":[0, 1e-1, 1],
      'random_state': [42]
    }

    eval_set = [(X_train, y_train), (X_val, y_val)]
    model = lgb.LGBMRegressor()
    clf = GridSearchCV(model, parameters)

    clf.fit(X_train, y_train, eval_set=eval_set,callbacks=[lgb.early_stopping(stopping_rounds=20,verbose=False)])


    model = lgb.LGBMRegressor(**clf.best_params_)
    # if True:
    #     plot_val_loss(X_train,y_train,eval_set,clf.best_params_)
    forecast=[]

    length=len(X)-offset
    for i in range(offset, len(X), batch):
        # Predict on current batch
        end = min(i + batch, len(X))
        end_train = i-horizon+1
        X_train, X_test, y_train, y_test=X[i-(length):end_train].to_numpy(),X[i:end].to_numpy(),Y[i-(length):end_train].to_numpy(),Y[i:end].to_numpy()

        # split the data into training, validation, and test sets
        X_val,  X_train,y_val ,  y_train= train_test_split(X_train, y_train, train_size=0.2, random_state=0)

        eval_set = [(X_train, y_train), (X_val, y_val)]

        model.fit(X_train, y_train,eval_set=eval_set,callbacks=[lgb.early_stopping(stopping_rounds=20,verbose=False)])

        forecast.extend(model.predict(X_test))
        
    forecast=np.array(forecast).reshape(-1, 1)

    return forecast

def arima_ma(X,Y,offset,horizon,batch,batch_error):

    X_train, y_train, =X['y'][:offset-horizon].to_numpy(),Y[:offset-horizon].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,shuffle=False)

    p_values = range(0,6)
    d_values = range(0, 3)
    q_values = range(0, 2)
    best_score=np.inf
    fc=np.empty(shape=(len(X_val),1))
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                
                for i in range(len(X_val)):
                    model=ARIMA(endog=X_train,order=order).fit()
                    fc[i]=np.mean(model.forecast(horizon))
                    

                rsme=np.sqrt(mean_squared_error(y_val,fc))
                if rsme < best_score:
                    best_score, best_cfg = rsme, order

    batch=1
    # best_cfg=(2,2,0)
    forecast=[]
    for i in range(offset, len(X), batch):

        end_train = i-horizon
        X_train, y_train=X['y'][i-offset:end_train].to_numpy(),Y[i-offset:end_train].to_numpy(),

        model=ARIMA(endog=X_train,order=best_cfg).fit()

        forecast.extend([np.mean(model.forecast(horizon))])
        
    forecast=np.array(forecast)

    forecast = forecast.reshape(-1, 1)

    N=batch_error

    errors = np.empty(forecast.shape)

    errors[:N]=best_score

    y_test=Y[offset:].to_numpy()
    # calculate RMSE for each batch of N values
    for i in range(N, len(forecast), N):
        rmse = np.sqrt(np.mean((forecast[i:i+N] - y_test[i:i+N]) ** 2))
        errors[i:i+N]=rmse

    df = pd.DataFrame(errors)
    df.fillna(method='ffill', inplace=True)
    errors = df.to_numpy()

    return forecast,errors


def arima(X,Y,offset,horizon,batch,batch_error):

    
    X_train, y_train, =X['y'][:offset+1].to_numpy(),Y[:offset+1].to_numpy()


    # Define the p, d and q parameters
    p = range(0,10)
    d = range(0, 3)
    q = range(0, 10)
    # Generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))

    # Initialize the best parameters and the best AIC
    best_aic = np.inf
    best_pdq = None

    # Grid search
    for param in pdq:
        try:
            model = ARIMA(X_train, order=param)
            model_fit = model.fit()
            
            # If the current run of AIC is better than the best one so far, overwrite the best AIC and best pdq
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_pdq = param
        except:
            continue
    batch=1
    # best_cfg=(2,2,0)
    forecast=[]
    for i in range(offset, len(X), batch):

        end_train = i+1
        X_train, y_train=X['y'][i-offset:end_train].to_numpy(),Y[i-offset:end_train].to_numpy(),
        # X_train, y_train=X['y'][:end_train].to_numpy(),Y[:end_train].to_numpy(),

        model=ARIMA(endog=X_train,order=best_pdq).fit()

        forecast.extend([(model.forecast(horizon))])
        
    forecast=np.array(forecast)

    # forecast = forecast.reshape(-1, 1)

    # N=batch_error

    # errors = np.empty(forecast.shape)

    # errors[:N]=best_score

    # y_test=Y[offset:].to_numpy()
    # calculate RMSE for each batch of N values
    # for i in range(N, len(forecast), N):
    #     rmse = np.sqrt(np.mean((forecast[i:i+N] - y_test[i:i+N]) ** 2))
    #     errors[i:i+N]=rmse

    # df = pd.DataFrame(errors)
    # df.fillna(method='ffill', inplace=True)
    # errors = df.to_numpy()

    return forecast

def arima2(X,Y,offset,horizon,batch,batch_error):

    
    X_train, y_train, =X['y'][:offset+1].to_numpy(),Y[:offset+1].to_numpy()


    # Define the p, d and q parameters
    p = range(0,6)
    d = range(0, 3)
    q = range(0, 6)
    # Generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))

    # Initialize the best parameters and the best AIC
    best_aic = np.inf
    best_pdq = None

    # Grid search
    for param in pdq:
        try:
            model = ARIMA(X_train, order=param)
            model_fit = model.fit()
            
            # If the current run of AIC is better than the best one so far, overwrite the best AIC and best pdq
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_pdq = param
        except:
            continue
    batch=1
    # best_cfg=(2,2,0)
    forecast=[]
    for i in range(offset, len(X), batch):

        end_train = i+1
        X_train, y_train=X['y'][i-offset:end_train].to_numpy(),Y[i-offset:end_train].to_numpy(),
        # X_train, y_train=X['y'][:end_train].to_numpy(),Y[:end_train].to_numpy(),

        model=ARIMA(endog=X_train,order=best_pdq).fit()

        forecast.extend([(model.forecast(horizon))])
        
    forecast=np.array(forecast)

    # forecast = forecast.reshape(-1, 1)

    # N=batch_error

    # errors = np.empty(forecast.shape)

    # errors[:N]=best_score

    # y_test=Y[offset:].to_numpy()
    # calculate RMSE for each batch of N values
    # for i in range(N, len(forecast), N):
    #     rmse = np.sqrt(np.mean((forecast[i:i+N] - y_test[i:i+N]) ** 2))
    #     errors[i:i+N]=rmse

    # df = pd.DataFrame(errors)
    # df.fillna(method='ffill', inplace=True)
    # errors = df.to_numpy()

    return forecast

def arima3(X,Y,offset,horizon,batch,batch_error):

    
    X_train, y_train, =X['y'][:offset+1].to_numpy(),Y[:offset+1].to_numpy()


    # Define the p, d and q parameters
    p = range(0,6)
    d = range(0, 3)
    q = range(0, 6)
    # Generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))

    # Initialize the best parameters and the best AIC
    best_aic = np.inf
    best_pdq = None

    # Grid search
    for param in pdq:
        try:
            model = ARIMA(X_train, order=param)
            model_fit = model.fit()
            
            # If the current run of AIC is better than the best one so far, overwrite the best AIC and best pdq
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_pdq = param
        except:
            continue
    batch=1
    # best_cfg=(2,2,0)
    forecast=[]
    length=len(X)-offset
    for i in range(offset, len(X), batch):

        end_train = i+1
        X_train, y_train=X['y'][i-length:end_train].to_numpy(),Y[i-length:end_train].to_numpy(),
        # X_train, y_train=X['y'][:end_train].to_numpy(),Y[:end_train].to_numpy(),

        model=ARIMA(endog=X_train,order=best_pdq).fit()

        forecast.extend([(model.forecast(horizon))])
        
    forecast=np.array(forecast)

    return forecast


def arimabic(X,Y,offset,horizon,batch,batch_error):

    
    X_train, y_train, =X['y'][:offset+1].to_numpy(),Y[:offset+1].to_numpy()


    # Define the p, d and q parameters
    p = range(0,6)
    d = range(0, 3)
    q = range(0, 6)
    # Generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))

    best_bic = np.inf
    best_pdq = None

    # Grid search
    for param in pdq:
        try:
            model = ARIMA(X_train, order=param)
            model_fit = model.fit()
            
            # If the current run of BIC is better than the best one so far, overwrite the best BIC and best pdq
            if model_fit.bic < best_bic:
                best_bic = model_fit.bic
                best_pdq = param
        except:
            continue
    batch=1
    # best_cfg=(2,2,0)
    forecast=[]
    length=len(X)-offset
    for i in range(offset, len(X), batch):

        end_train = i+1
        X_train, y_train=X['y'][i-length:end_train].to_numpy(),Y[i-length:end_train].to_numpy(),
        # X_train, y_train=X['y'][:end_train].to_numpy(),Y[:end_train].to_numpy(),

        model=ARIMA(endog=X_train,order=best_pdq).fit()

        forecast.extend([(model.forecast(horizon))])
        
    forecast=np.array(forecast)

    return forecast


def arimahqic(X,Y,offset,horizon,batch,batch_error):

    
    X_train, y_train, =X['y'][:offset+1].to_numpy(),Y[:offset+1].to_numpy()


    # Define the p, d and q parameters
    p = range(0,6)
    d = range(0, 3)
    q = range(0, 6)
    # Generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))

    # Initialize the best parameters and the best information criterion (AIC, BIC, HQIC)
    best_criterion_value = np.inf
    best_pdq = None

    # Grid search
    for param in pdq:
        try:
            # Fit ARIMA model
            model = ARIMA(X_train, order=param)
            model_fit = model.fit()
            
            # Choose the desired information criterion
            # For example, to use AIC:
            # criterion_value = model_fit.aic
            # For BIC:
            # criterion_value = model_fit.bic
            # For HQIC:
            criterion_value = model_fit.hqic

            # Update the best parameters if the current criterion value is better
            if criterion_value < best_criterion_value:
                best_criterion_value = criterion_value
                best_pdq = param
        except:
            continue
    batch=1
    # best_cfg=(2,2,0)
    forecast=[]
    length=len(X)-offset
    for i in range(offset, len(X), batch):

        end_train = i+1
        X_train, y_train=X['y'][i-length:end_train].to_numpy(),Y[i-length:end_train].to_numpy(),
        # X_train, y_train=X['y'][:end_train].to_numpy(),Y[:end_train].to_numpy(),

        model=ARIMA(endog=X_train,order=best_pdq).fit()

        forecast.extend([(model.forecast(horizon))])
        
    forecast=np.array(forecast)

    return forecast

def arimarmse(X,Y,offset,horizon,batch,batch_error):

    
    X_t, y_t, =X['y'][:offset+1].to_numpy(),Y[0][:offset+1].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size=0.1, random_state=0,shuffle=False)

    best_score=np.inf
    val_size=len(X_val)

    # Define the p, d and q parameters
    p = range(0,6)
    d = range(0, 3)
    q = range(0, 6)
    # Generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))

    # Initialize the best parameters and the best AIC
    best_pdq = None

    
    # Grid search
    for param in pdq:
        try:

            model=ARIMA(endog=X_train,order=param).fit()
            fc=(model.forecast(val_size))
                    

            rsme=np.sqrt(mean_squared_error(y_val,fc))
            if rsme < best_score:
                best_score, best_pdq = rsme, param
        except:
            continue
    batch=1
    # best_cfg=(2,2,0)
    forecast=[]
    length=len(X)-offset
    for i in range(offset, len(X), batch):

        end_train = i+1
        X_train, y_train=X['y'][i-length:end_train].to_numpy(),Y[i-length:end_train].to_numpy(),
        # X_train, y_train=X['y'][:end_train].to_numpy(),Y[:end_train].to_numpy(),

        model=ARIMA(endog=X_train,order=best_pdq).fit()

        forecast.extend([(model.forecast(horizon))])
        
    forecast=np.array(forecast)

    return forecast

def arima4(X,Y,offset,horizon,batch,batch_error):

    length=len(X)-offset
    X_train, y_train, =X['y'][offset-length:offset+1].to_numpy(),Y[offset-length:offset+1].to_numpy()


    # Define the p, d and q parameters
    p = range(0,6)
    d = range(0, 3)
    q = range(0, 6)
    # Generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))

    # Initialize the best parameters and the best AIC
    best_aic = np.inf
    best_pdq = None

    # Grid search
    for param in pdq:
        try:
            model = ARIMA(X_train, order=param)
            model_fit = model.fit()
            
            # If the current run of AIC is better than the best one so far, overwrite the best AIC and best pdq
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_pdq = param
        except:
            continue
    batch=1
    # best_cfg=(2,2,0)
    forecast=[]
    
    for i in range(offset, len(X), batch):

        end_train = i+1
        X_train, y_train=X['y'][i-length:end_train].to_numpy(),Y[i-length:end_train].to_numpy(),
        # X_train, y_train=X['y'][:end_train].to_numpy(),Y[:end_train].to_numpy(),

        model=ARIMA(endog=X_train,order=best_pdq).fit()

        forecast.extend([(model.forecast(horizon))])
        
    forecast=np.array(forecast)

    # forecast = forecast.reshape(-1, 1)

    # N=batch_error

    # errors = np.empty(forecast.shape)

    # errors[:N]=best_score

    # y_test=Y[offset:].to_numpy()
    # calculate RMSE for each batch of N values
    # for i in range(N, len(forecast), N):
    #     rmse = np.sqrt(np.mean((forecast[i:i+N] - y_test[i:i+N]) ** 2))
    #     errors[i:i+N]=rmse

    # df = pd.DataFrame(errors)
    # df.fillna(method='ffill', inplace=True)
    # errors = df.to_numpy()

    return forecast

# define a custom function to calculate the trend of a row
def calculate_trend(row):
    # calculate the slope of the row using linear regression

    x = np.arange(len(row))
    res= np.polyfit(x, row, 1)[0]


    return res


def calculate_metrics(y_hat,y,y_hat_trend,y_trend,train_test='test'):
               
        def calculate_directional_accuracy(real_forecast, predicted_forecast):
            # Initialize variables to count correct direction forecasts and total forecasts
            correct_direction_forecasts = 0
            total_forecasts = 0
            
            # Loop through the forecasts and compare the spread direction
            for real, predicted in zip(real_forecast, predicted_forecast):
                # Determine the spread direction for each forecast
                if real == predicted:
                    correct_direction_forecasts += 1  # Correct "Up" forecast

                total_forecasts += 1

            return correct_direction_forecasts/total_forecasts*100

        mse=mean_squared_error(y, y_hat)
        rmse = np.sqrt(mean_squared_error(y, y_hat))
        mae = mean_absolute_error(y, y_hat)
        r2 = r2_score(y, y_hat)
        smape = 100 * np.mean(2 * np.abs(y_hat - y) / (np.abs(y) + np.abs(y_hat)))
        
        direction_acc=calculate_directional_accuracy(y_trend,y_hat_trend)
        y_trend = np.where(y_trend > 0, 1, 0)
        y_hat_trend = np.where(y_hat_trend > 0, 1, 0)
        cm=confusion_matrix(y_trend, y_hat_trend)

        if train_test=='test':     
            return {'Test MSE': mse,'Test RMSE': rmse, 'Test MAE': mae, 'Test R2': r2, 'Test sMAPE': smape, "Test DirectionalACC": direction_acc }
        return {'Train RMSE': rmse, 'Train MAE': mae, 'Train R2': r2, 'Train sMAPE': smape,"Train DirectionalACC": direction_acc}





def plot_components_and_spread(asset_A_prices, asset_B_prices,c1,c2):
    spread = c1-c2
    names = spread.name.split("_")
    asset_A_prices = pd.Series(asset_A_prices)
    asset_B_prices = pd.Series(asset_B_prices)
    position=historical_returns(spread,entry_l=-2,entry_s=2)
    # Calculate the z-scored spread (normalized spread)
    mean_spread = np.mean(spread)
    std_spread = np.std(spread)
    zspread=(spread-mean_spread)/std_spread

    # Define the positions on the x-axis
    index = range(0, len(asset_A_prices))

    # Define your custom x-axis labels
    x_labels = {0: '2006', 252: '2007', 504: '2008', 756: '2009', 1005: '2010'}

    # plt.figure(figsize=(12, 6))

    # plt.xlabel("Year")
    # plt.ylabel("Price ($)")

    # # Plot z-spread
    # plt.plot(index, asset_A_prices, label=names[0])
    # # # Plot position
    # plt.plot(index, asset_B_prices, label=names[1])


    label_indices = np.linspace(0, len(zspread)-1, 10).astype(int)
    fig, axs = plt.subplots(2, 1, sharex=True)  # Create two subplots sharing x axis

    # Plot z-spread
    axs[0].plot(zspread, label='Normalized Spread')
    axs[0].set_ylabel("Normalized Spread")
    axs[0].axhline(y=2, color='black', linestyle='--', label='Short Threshold')
    axs[0].axhline(y=-2, color='black', linestyle=':', label='Long Threshold')
    axs[0].legend()

    # Plot position
    axs[1].step(range(len(position)), position, label='Position', color='purple', where='post')
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Position")
    axs[1].legend()

    # plt.legend()
    # plt.subplots_adjust(bottom=0.2)

    # # Add a grid
    # plt.grid(True)

    # sns.set_theme()
    # sns.set_style("dark")
    # sns.set_context("paper", font_scale = 1)

    # # Set x-ticks at the label indices
    plt.xticks(label_indices, rotation=45)
    plt.xlim(min(label_indices), max(label_indices))
    # plt.show()

    # plt.xlabel("Year")
    # plt.ylabel("Normalized Spread")

    # # Plot z-spread
    # plt.plot(index, zspread, label='Normalized Spread')


    # # Plot position
    # plt.plot(index, position, label='Position', color='purple')

    # # Plot short and long thresholds
    # plt.axhline(y=2, color='black', linestyle='--', label='Short Threshold')
    # plt.axhline(y=-2, color='black', linestyle=':', label='Long Threshold')

    plt.legend()
    plt.subplots_adjust(bottom=0.2)

    # Add a grid
    plt.grid(True)

    sns.set_theme()
    sns.set_style("dark")
    sns.set_context("paper", font_scale = 1)

    # Create a custom function to format the x-ticks
    # def format_func(value, tick_number):
    #     return x_labels.get(value, '')
    # tick_positions = list(x_labels.keys())
    # plt.xticks(tick_positions, x_labels.values())

    # Use FuncFormatter to apply the custom function
    # plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

    # plt.xlim(min(index), max(index))

    # Save the figure with high resolution
    # plt.savefig("portfolio_retssp.png", dpi=1000)

    plt.show()

def plot_forecasts(real,fc,horizon,model):
    import seaborn as sns
    import matplotlib.ticker as ticker
    # # Define your custom x-axis labels
    # months = ['Mar', 'Jun',  'Sep', 'Oct', 'Nov', 'Dec','Jan']
    # x_labels = {int((i+1)*len(real)/3): months[i] for i in range(0, 4)}

  # Define the positions on the x-axis
    index = range(0, len(real[:-horizon]))
    plt.figure(figsize=(12, 6))
    plt.plot(index,real[:-horizon], label='Real', color='blue')
    plt.plot(index,fc[:-horizon], label=model, color='green')


    # Add labels, legend, and grid
    plt.xlabel('Day')
    plt.ylabel('Value')
    # plt.title('Price with Moving Averages')

    plt.legend()
    plt.subplots_adjust(bottom=0.2)
    # Add a grid
    plt.grid(True)

    sns.set_theme()
    sns.set_style("dark")
    sns.set_context("paper", font_scale =2)

    # def format_func(value, tick_number):
    #     return x_labels.get(value, '')
    # tick_positions = list(x_labels.keys())
    # plt.xticks(tick_positions, x_labels.values())

    # Use FuncFormatter to apply the custom function
    # plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

    plt.xlim(min(index), max(index))
    plt.title('Predictions of forecasting model')
    # Show the plot
    plt.show()



def feature_analysis(features,target,offset):

    X_train=features
    y_train=target
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,shuffle=False)
    parameters = {
      'n_estimators': [400],
      'learning_rate': [0.001,0.01, 0.1],
      'max_depth': [8],
      'num_leaves': [2, 8, 32,256,1024],
      "reg_alpha":[0.1], 
      "reg_lambda":[1.0],
      'random_state': [42]
    }

    eval_set = [(X_train, y_train), (X_val, y_val)]
    model = lgb.LGBMRegressor(verbose=-1,reg_alpha=0.1, reg_lambda=1.0)

    model.fit(X_train, y_train)

    ax = lgb.plot_importance(model, max_num_features=40, figsize=(15,15))
    plt.show()


def plot_val_loss(X_train,y_train,eval_set,best_params):
    # Fit the model with the best parameters
    model = lgb.LGBMRegressor(**best_params, verbose=-1)
    model.fit(X_train, y_train, eval_set=eval_set,callbacks=[lgb.early_stopping(stopping_rounds=20)])

    # Get the training and validation scores
    training_scores = model.evals_result_['training']['l2']
    validation_scores = model.evals_result_['valid_1']['l2']

    # Plot the training and validation scores
    plt.figure(figsize=(10, 5))
    plt.plot(training_scores, label='Training')
    plt.plot(validation_scores, label='Validation')
    plt.title('Training and Validation Scores')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.legend()
    plt.show()