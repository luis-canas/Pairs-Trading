
import numpy as np
import statsmodels.api as sm

from utils.utils import normal
from pymoo.core.problem import ElementwiseProblem

import math

from pymoo.termination.default import DefaultSingleObjectiveTermination


from utils.utils import load_args



import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter


from pymoo.algorithms.soo.nonconvex.ga import GA

from pymoo.optimize import minimize

from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
import pickle
from os.path import exists

from scipy.stats import norm

# Constants
NB_TRADING_DAYS = 252  # 1 year has 252 trading days

LONG_SPREAD = 1
SHORT_SPREAD = -1
CLOSE_POSITION = 0


class CInt(ElementwiseProblem):

    def __init__(self, returns, spreads,window,objectives=["SR"],cardinality=10):

        # spread and components price series
        self.returns = returns.pct_change()[window:].dropna()
        self.N_DATAPOINTS=self.returns.shape[0]
        self.N_PAIRS=self.returns.shape[1]
        self.N_VARS=cardinality*2
        self.cov=np.cov(self.returns.T)
        self.sigma = np.array(self.cov, ndmin=2)
        self.mu=np.array(self.returns.mean(), ndmin=2)


        self.objectives = objectives

     
        xl=[0]*(cardinality)
        xu=[1]*(cardinality)
        xl=np.concatenate((xl,[0]*(cardinality)))
        xu=np.concatenate((xu,[self.N_PAIRS-1]*(cardinality)))

        super().__init__(n_var=self.N_VARS,
                         n_obj=len(objectives),
                         n_constr=0,
                         xl=xl,
                         xu=xu,
                         vtype=float)

    def _repair_constraints(self,w):
        return normal(w)
    
    def _evaluate(self, x, out, *args, **kwargs):

        
        weight,asset=np.reshape(x,newshape=(2,-1))
        asset=np.around(asset).astype(int)
        w=np.zeros((self.N_PAIRS,1))
        w[asset]=weight.reshape(-1,1)

        w=self._repair_constraints(w)

        objectives=self.objectives

        f = np.zeros(len(objectives))

        index = 0

        if "ROI" in objectives:
            roi =  (self.mu @ w )
            f[index] = (-roi)
            index += 1

        if "SR" in objectives:
            ret=(self.mu @ w )
            risk=np.sqrt(w.T @ self.cov @ w)
            # Calculate the Sharpe Ratio
            if risk != 0:
                sharpe_ratio = ret / risk
            else:
                sharpe_ratio = 0
            
            f[index] = (-sharpe_ratio)
            index += 1
        if "MV" in objectives:
            risk= w.T @ self.cov @ w

            f[index] = (risk)
            index += 1
        
        out["F"] = f





def ga_weights(returns,spreads,sector,window,pop=100,gen=2000,objectives=["SR"],**kwargs):

    verbose=False
    plot=False

    args=load_args("GA")
    cv=args.get('cv')
    mt=args.get('mt')
    gen=args.get('gen')
    seed=args.get('seed')
    callback=args.get('callback')
    percentage=args.get('percentage')

    compute_w=False

    file='results/'+'weights.pkl'

    isExist = exists(file)
    obj = '_'.join(objectives)
    cardinality=math.ceil(returns.shape[1]*percentage)

    string=returns.index[0]+'_'+returns.index[-1]+obj+str(cardinality)+str(pop)+str(gen)+sector

    if isExist:
        # Load the data from a pickle file
        with open(file, 'rb') as f:
            weights_cint = pickle.load(f)
    else:
        weights_cint={}

    if compute_w or (string not in weights_cint):

        # Build genetic algorithm
        algorithm = GA(pop_size=pop,
                            crossover=UniformCrossover(prob=cv),
                            mutation=PolynomialMutation(prob=mt),
                            eliminate_duplicates=True)
        
        # Get objective function
        opt = CInt(returns=returns, spreads=spreads,window=window,objectives=objectives,cardinality=cardinality)

        termination =  DefaultSingleObjectiveTermination(xtol=0,cvtol=0.0,ftol=1e-5,period=callback,n_max_gen=gen,n_max_evals=pop*gen)
        
        # Optimize patterns
        results = minimize(opt, algorithm, termination,seed=seed, save_history=True, verbose=verbose)

        RX,RF=np.array(results.X),np.array(results.F)
        weights_cint.update({string: [RX,RF]})

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

        
        # opt = fitness.flatten()
        # port= np.reshape(portfolio, (portfolio.shape[0] * portfolio.shape[1], portfolio.shape[2]))
        directory=''


        if plot and False:
            fs=25

            plt.subplots(figsize=(15, 10))
            sns.set_theme()
            sns.set_style("white")
            sns.set_context("paper", font_scale=5)
            plt.plot(n_evals, opt, linestyle='--',color='black')

            # Increase font size for the x-axis label
            plt.xlabel('Evaluation', fontsize=fs)

            # Increase font size for the y-axis label
            plt.ylabel('Fitness', fontsize=fs)

            # Increase font size for the tick labels on both axes
            plt.xticks(fontsize=fs)
            plt.yticks(fontsize=fs)
            plt.tight_layout()
            plt.savefig(directory+"convergence.png")
            plt.show()

        if plot:
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
            plt.savefig(directory+"portfolio"+str(percentage*100)+".png")
            plt.show()

        if plot:

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
                    # Build genetic algorithm
                    algorithm = GA(pop_size=pop,
                                        crossover=UniformCrossover(prob=cv),
                                        mutation=PolynomialMutation(prob=mt),
                                        eliminate_duplicates=True)
                    
                    # Get objective function
                    opt = CInt(returns=returns, spreads=spreads,window=window,objectives=objectives,cardinality=cardinality)

                    termination =  DefaultSingleObjectiveTermination(xtol=0,cvtol=0.0,ftol=1e-7,period=gen,n_max_gen=gen,n_max_evals=pop*gen)
                    
                    # Optimize patterns
                    results = minimize(opt, algorithm, termination,seed=8, save_history=True, verbose=verbose)

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
            if plot:
                
                                # Set font scale globally
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
                    plt.savefig(directory + returns.index[0][:4] + filename)
                    # plt.show()


                

        # y to a pickle file
        with open(file, 'wb') as f:
            pickle.dump(weights_cint, f)
    
    else:
        [RX,RF]=weights_cint[string]

    I=np.argmax(RF)
    res=RX if RX.ndim==1 else RX[I]

    x,asset=np.reshape(res,newshape=(2,-1))
    
    asset=np.around(asset).astype(int)

    weights=np.zeros(returns.shape[1])
    
    weights[asset]=x

    weights= np.array(weights/np.sum(weights), ndmin=2)

    return weights


def clean_weights(weights,th=0.0001,uni=False):
    # Substitute all values in the array lower than the threshold for 0

    subs=1 if uni else weights

    arr = np.where(weights < th, 0, subs)
    arr_sum=np.sum(arr)
    
    
    return np.ravel(arr/arr_sum)


def uniform_portfolio(N):

    return [1/N]*N
