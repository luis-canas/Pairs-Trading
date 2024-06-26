
import numpy as np

from utils.utils import normal,portfolio_plots,load_args
from pymoo.core.problem import ElementwiseProblem

import math

from pymoo.termination.default import DefaultSingleObjectiveTermination

from pymoo.algorithms.soo.nonconvex.ga import GA

from pymoo.optimize import minimize

from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
import pickle
from os.path import exists

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





def ga_weights(returns,spreads,sector,window,pop=100,gen=1000,objectives=["SR"],**kwargs):

    verbose=False
    plot=False

    args=load_args("GA")
    cv=args.get('cv')
    mt=args.get('mt')
    gen=args.get('gen')
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
        optimization = CInt(returns=returns, spreads=spreads,window=window,objectives=objectives,cardinality=cardinality)

        termination =  DefaultSingleObjectiveTermination(xtol=0,cvtol=0.0,ftol=1e-5,period=100,n_max_gen=gen,n_max_evals=pop*gen)
        
        # Optimize patterns
        results = minimize(optimization, algorithm, termination,seed=8, save_history=True, verbose=verbose)

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

        if plot:
            portfolio_plots(opt,port,returns,obj,pop,gen,sector,window,percentage,algorithm,optimization)              

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


def clean_weights(weights,th=1e-4,uni=False):
    # Substitute all values in the array lower than the threshold for 0

    subs=1 if uni else weights

    arr = np.where(weights < th, 0, subs)
    arr_sum=np.sum(arr)
    
    
    return np.ravel(arr/arr_sum)


def uniform_portfolio(N):

    return [1/N]*N
