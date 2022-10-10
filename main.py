from class_Pairs import Pairs
from class_Trader import Trader
from class_Portfolio import Portfolio


portfolio=Portfolio()

portfolio.portfolio1()
data_train,data_val=portfolio.get_train_val()



# Train Tickers
cointegration=Pairs(data_train)
selected_pairs=cointegration.get_pairs()
print(selected_pairs)

# Validate Tickers


