

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.model_selection import GridSearchCV,train_test_split

import numpy as np

import itertools


from sklearn.metrics import mean_squared_error,confusion_matrix


import lightgbm as lgb


import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Ignore ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
# Ignore specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


NB_TRADING_DAYS = 252

LONG_SPREAD = 1
SHORT_SPREAD = -1
CLOSE_POSITION = 0


def lightgbm(X,Y,offset,horizon,batch):

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









def arima(X,Y,offset,horizon):

    
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

    forecast=[]
    length=len(X)-offset
    for i in range(offset, len(X),1):

        end_train = i+1
        X_train, y_train=X['y'][i-length:end_train].to_numpy(),Y[i-length:end_train].to_numpy(),

        model=ARIMA(endog=X_train,order=best_pdq).fit()

        forecast.extend([(model.forecast(horizon))])
        
    forecast=np.array(forecast)

    return forecast








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


