
import math
import random
import numpy as np


NB_TRADING_DAYS=252

  
    
import gymnasium as gym
import numpy as np
import pandas as pd

class PairsTradingEnvironment(gym.Env):
    
    def __init__(self, spread=1,c1=1,c2=1 ,FIXED_VALUE=1000, commission=0.08, market_impact=0.2, short_loan=1, b=1):
        super(PairsTradingEnvironment, self).__init__()

        self.FIXED_VALUE = FIXED_VALUE

        # define trading costs
        self.fixed_costs_per_trade = (
            commission + market_impact) / 100  # remove percentage
        self.short_costs_per_day = FIXED_VALUE * \
            (short_loan / NB_TRADING_DAYS) / 100  # remove percentage
        
        self.b = b

        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Close

        # Calculate the spread from data (assuming c1 and c2 are the first two columns)
        self.spread = spread
        self.c1 = c1
        self.c2 = c2

        # Define the observation space (state space)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(spread.columns),))

        self.current_step = 0
        self.stocks_in_hand = np.zeros(2)  # Number of stocks in hand (positions)
        self.cash_in_hand = self.FIXED_VALUE  # Starting balance
        self.portfolio_value = self.FIXED_VALUE  # Portfolio value (cash + stocks)
        self.position=0
        self.done=False
        

    def render(self,mode='human'):
        pass
    
    def reset(self, seed=None, options=None):

        self.current_step = 0
        self.stocks_in_hand = np.zeros(2)
        self.cash_in_hand = self.FIXED_VALUE
        self.portfolio_value = self.FIXED_VALUE
        self.position=0
        self.done=False

        # Reset the environment and return the initial observation/state
        return self._get_observation(),{'portfolio_value': self.portfolio_value}

    def _get_observation(self):
        # Get the current state/observation from the data

        return self.spread.iloc[self.current_step]

    def _calculate_reward(self):


        # Update portfolio value based on the current state
        self.portfolio_value = self.cash_in_hand + self.stocks_in_hand[0] * self.c1[self.current_step] + self.stocks_in_hand[1] * self.c2[self.current_step]
        
        return self.portfolio_value

    def _apply_transaction_costs(self, value_to_buy):
        # Apply transaction costs (commission and market impact)
        self.cash_in_hand -= 2*self.fixed_costs_per_trade*value_to_buy

    def step(self, action):
        # Execute the action and get the next state, reward, done flag, and additional info
        sale_value=0
        value_to_buy = min(self.FIXED_VALUE, max(self.cash_in_hand,0))

        if self.position!=action:

            # at the beginning of the day we still have the cash we had the day before
            sale_value = self.stocks_in_hand[0] * self.c1[self.current_step] + self.stocks_in_hand[1] * self.c2[self.current_step]
            # 2 closes, so 2*transaction costs
            self.cash_in_hand += sale_value * (1 - 2*self.fixed_costs_per_trade)

            # both positions were closed
            self.stocks_in_hand[0] = self.stocks_in_hand[1] = 0


            if action == 1:  # LONG SPREAD
                self.cash_in_hand += value_to_buy 
                self.stocks_in_hand[1] = -value_to_buy  / self.c2[self.current_step]

                self.cash_in_hand += -value_to_buy 
                self.stocks_in_hand[0] = value_to_buy / self.c1[self.current_step]
                self._apply_transaction_costs(value_to_buy)

            elif action == 2:  # SHORT SPREAD
                self.cash_in_hand += value_to_buy 
                self.stocks_in_hand[0] = -value_to_buy  / self.c1[self.current_step]

                self.cash_in_hand += -value_to_buy 
                self.stocks_in_hand[1] = value_to_buy / self.c2[self.current_step]
                self._apply_transaction_costs(value_to_buy)


            self.position=action

            # Apply short rental costs if there's an open position
            if self.stocks_in_hand[0] != 0 or self.stocks_in_hand[1] != 0:
                self.cash_in_hand -= self.short_costs_per_day

        info = {'portfolio_value': self.portfolio_value}
        
        current_value=self.portfolio_value
        reward = (self._calculate_reward()/self.FIXED_VALUE)*100

        self.current_step += 1
        obs=self._get_observation()
        self.done = True if self.current_step == len(self.spread)-1 else False

        return obs,reward,self.done,False,info




import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class Actions(Enum):
    Sell = 0
    Buy = 1


class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class TradingEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size, render_mode=None):
        assert df.ndim == 2

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = spaces.Box(low=-INF, high=INF, shape=self.shape, dtype=np.float64)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._terminated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def _get_info(self):
        return dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._terminated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}

        info = self._get_info()
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self._terminated = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._terminated = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == "human":
            self._render_frame()

        return observation, step_reward, self._terminated, False, info


    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]


    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _render_frame(self):
        self.render()

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)


    def render_all(self, title=None):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        if title: plt.title(title)
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
        
        
    def close(self):
        plt.close()


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()


    def _process_data(self):
        raise NotImplementedError


    def _calculate_reward(self, action):
        raise NotImplementedError


    def _update_profit(self, action):
        raise NotImplementedError


    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
    

class ForexEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, unit_side='left', render_mode=None):
        assert len(frame_bound) == 2
        assert unit_side.lower() in ['left', 'right']

        self.frame_bound = frame_bound
        self.unit_side = unit_side.lower()
        super().__init__(df, window_size, render_mode)

        self.trade_fee = 0.0003  # unit


    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features


    def _calculate_reward(self, action):
        step_reward = 0  # pip

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Short:
                step_reward += -price_diff * 10000
            elif self._position == Positions.Long:
                step_reward += price_diff * 10000

        return step_reward


    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._terminated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self.unit_side == 'left':
                if self._position == Positions.Short:
                    quantity = self._total_profit * (last_trade_price - self.trade_fee)
                    self._total_profit = quantity / current_price

            elif self.unit_side == 'right':
                if self._position == Positions.Long:
                    quantity = self._total_profit / last_trade_price
                    self._total_profit = quantity * (current_price - self.trade_fee)


    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            current_price = self.prices[current_tick - 1]
            last_trade_price = self.prices[last_trade_tick]

            if self.unit_side == 'left':
                if position == Positions.Short:
                    quantity = profit * (last_trade_price - self.trade_fee)
                    profit = quantity / current_price

            elif self.unit_side == 'right':
                if position == Positions.Long:
                    quantity = profit / last_trade_price
                    profit = quantity * (current_price - self.trade_fee)

            last_trade_tick = current_tick - 1

        return profit
    
class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size, render_mode)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit


    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features


    def _calculate_reward(self, action):
        step_reward = 0

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward += price_diff

        return step_reward


    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._terminated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price


    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
    
class PairsEnv(TradingEnv):

    def __init__(self, df1, df2, window_size, frame_bound, render_mode=None, fixed_costs_per_trade=0.01, short_costs_per_day=0.01):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.df1 = df1
        self.df2 = df2
        super().__init__(df1, window_size, render_mode)

        self.fixed_costs_per_trade = fixed_costs_per_trade
        self.short_costs_per_day = short_costs_per_day


    def _process_data(self):
        prices1 = self.df1.loc[:, 'Close'].to_numpy()
        prices2 = self.df2.loc[:, 'Close'].to_numpy()

        prices1[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices1 = prices1[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        
        prices2[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices2 = prices2[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        prices = prices1 / prices2
        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features


    def _calculate_reward(self, action):
        step_reward = 0

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True
            step_reward = 2*self.fixed_costs_per_trade 

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward = price_diff-self.short_costs_per_day
            if self._position == Positions.Short:
                step_reward = -price_diff-self.short_costs_per_day
        return step_reward


    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True
            self._total_profit -=2*self.fixed_costs_per_trade 

        if trade or self._terminated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            spread_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                self._total_profit += spread_diff-self.short_costs_per_day

            if self._position == Positions.Short:
                self._total_profit += -spread_diff-self.short_costs_per_day


    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit