
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

        termination =  DefaultSingleObjectiveTermination(xtol=0,cvtol=0.0,ftol=1e-7,period=callback,n_max_gen=gen,n_max_evals=pop*gen)
        
        # Optimize patterns
        results = minimize(opt, algorithm, termination,seed=8, save_history=True, verbose=verbose)

        RX,RF=np.array(results.X),np.array(results.F)
        weights_cint.update({string: [RX,RF]})

        n_evals = np.array([e.evaluator.n_eval for e in results.history]).reshape(-1,1)
        opt = -np.array([e.opt[0].F for e in results.history]).reshape(-1,1)
        port = np.array([e.opt[0].X for e in results.history])

        if plot:
            plt.title("Convergence")
            plt.plot(n_evals, opt, linestyle='--')
            plt.xlabel('Evaluation')
            plt.ylabel('Fitness')
            plt.show()

        if plot:
            rets = returns.pct_change()[window:].dropna()
            cov=np.cov(returns.T)
            mu=np.array(rets.mean(), ndmin=2)
            ro=len(port)
            co=2
            ret_risk=np.zeros(shape=(ro,co))
            for n in range(ro):
                x,asset=np.reshape(port[n],newshape=(2,-1))
                asset=np.around(asset).astype(int)
                weights=np.zeros(returns.shape[1])
                weights[asset]=x
                weights= np.array(weights/np.sum(weights), ndmin=2)
                daily_ret=(mu @ weights.T)
                daily_risk=np.sqrt(weights @ cov @ weights.T)
                ret_risk[n]= ((daily_ret + 1) ** NB_TRADING_DAYS - 1)*100, (daily_risk * np.sqrt(NB_TRADING_DAYS))*100
            # Set the seaborn theme
            sns.set_theme()
            sns.set_style("dark")
            sns.set_context("paper", font_scale=1.5)
            # Plot return versus risk
            fig, ax = plt.subplots(figsize=(10, 6))
            # Define a custom colormap with a smooth gradient from purple to blue to green to yellow
            colors = ['#800080', '#0000FF', '#00FF00', '#FFFF00']  # Purple to blue to green to yellow gradient
            cmap_name = 'custom_gradient'
            cm = LinearSegmentedColormap.from_list(cmap_name, colors)
            # Create a gradient array based on the position in the array
            gradient = np.linspace(0, len(ret_risk[:, 1]),len(ret_risk[:, 1]))
            # Plot the first set of scatter points with the custom gradient
            sc = ax.scatter(ret_risk[:, 1], ret_risk[:, 0], marker='o', c=gradient, alpha=0.6, cmap=cm)
            # Add a label to the second set of scatter points
            ax.text(ret_risk[-1, 1], ret_risk[-1, 0], 'Optimal Portfolio', fontsize=12, ha='right', va='bottom', color='black')
            # Customize plot labels and title
            ax.set_xlabel('Expected Risk (%)')
            ax.set_ylabel('Expected Return (%)')
            ax.set_title('Portfolio by generation')
            ax.grid(True)
            # Create a ScalarMappable object for the custom gradient
            sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])  # An empty array is needed
            # Create a color bar for the custom gradient
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Generation')
            # Create a custom formatter for the color bar ticks to multiply by gen
            formatter = FuncFormatter(lambda x, _: '{:.0f}'.format(x * gen))
            cbar.ax.yaxis.set_major_formatter(formatter)
            # Show the plot
            plt.show()
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