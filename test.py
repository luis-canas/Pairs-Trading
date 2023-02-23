import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from operator import itemgetter
from statsmodels.api import OLS
from statsmodels.tsa.stattools import coint, adfuller
from hurst import compute_Hc as hurst_exponent
from scipy.stats import zscore

from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize


from class_Pairs import Pairs
from class_Trader import Trader
from class_History import History



import pandas as pd
import numpy as np
import seaborn as sns

from pandas_datareader import data
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
import pandas as pd
from datetime import datetime
import yfinance as yf
import pickle
import matplotlib.pyplot as plt


start="2005-01-01"
end="2010-01-01"



tickers = ['KO', 'PEP' ]

datas=yf.download(tickers, start=start, end=end)['Close']

dates=datas._stat_axis.date
plt.figure(figsize=(10,6), tight_layout=True)
#plotting
plt.plot(datas, linewidth=2)
#customization
# plt.xticks([2017, 2018, 2019, 2020, 2021])
plt.xlabel('Year')
plt.ylabel('Price (USD)')

print(datas)
plt.rc('legend', fontsize=16) 
plt.legend(tickers)
  

signal2=datas[tickers[1]]
signal1=datas[tickers[0]]
beta = OLS(signal2, signal1).fit().params[0]
spread = signal2-beta*signal1
result = coint(signal1, signal2)
score = result[0]
pvalue = result[1]
hurst, _, _ = hurst_exponent(spread)



plt.figure(figsize=(10,6), tight_layout=True)
normalized_spread = zscore(spread)
plt.plot(normalized_spread, linewidth=2)
standard_devitation = np.std(normalized_spread)

plt.axhline(standard_devitation, linestyle='--',color='green')
plt.axhline(-standard_devitation, linestyle='--',color='green')
plt.axhline(2*standard_devitation,linestyle= '--', color='red')
plt.axhline(-2*standard_devitation, linestyle= '--',color='red')


stop_loss=2*standard_devitation
entry=standard_devitation
close=0

# Compute the z score for each day
zs = normalized_spread
returns=[]
open_position=False
initial_value=0
p=0
l=0
loss=False
for i in range(len(normalized_spread)):
    if open_position:
        if zs[i]>stop_loss and not below or  zs[i]<-stop_loss and below:
            open_position=False
            loss=True
            plt.plot(dates[i],2*standard_devitation*normalized_spread[i]/np.abs(normalized_spread[i]),marker="^",color='red',markersize=15)
        elif zs[i]>close and below or zs[i] < close and not below:   
            open_position=False             
            plt.plot(dates[i],0,marker="^",color='blue',markersize=15)

        else:
            pass  
    else:
        if (zs[i]>entry and zs[i]>0  or zs[i]<-entry and zs[i]<0) and not loss:
            open_position=True
            below=False if  zs[i]>0 else True
            s=normalized_spread[i]
            
            plt.plot(dates[i],standard_devitation*normalized_spread[i]/np.abs(normalized_spread[i]),marker="^",color='green',markersize=15,label="Entry")
        if (zs[i]<entry and zs[i]>0  or zs[i]>-entry and zs[i]<0) and loss:
            open_position=True
            below=False if  zs[i]>0 else True
            loss=False
            s=normalized_spread[i]
            
            plt.plot(dates[i],standard_devitation*normalized_spread[i]/np.abs(normalized_spread[i]),marker="^",color='green',markersize=15,label="Entry")


from matplotlib.legend_handler import HandlerTuple
handler_map={tuple: HandlerTuple(ndivide=None)}
print(handler_map)
plt.legend(['Spread', '_nolegend_','_nolegend_','_nolegend_','_nolegend_','Open Position','_nolegend_','_nolegend_','Close Position','_nolegend_','_nolegend_','_nolegend_','_nolegend_','_nolegend_','Stop Loss'],
               handler_map={tuple: HandlerTuple(ndivide=None)})


# entry= np.ma.masked_equal(returns, 3)
# x=normalized_spread[entry]
# print(x)
# print(np.count_nonzero(entry))
# plt.plot(x,'s',marker="^")
plt.xlabel('Year')
plt.ylabel('Standard Score')
plt.show()

# plt.show()