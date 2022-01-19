"""
This file contains an m-graph that does not have a valid m-adjustment set according to the M-adjustment criterion proposed
by Sadaati and Tian. However, the causal effect is recoverable via the IPW method in this case.

After reweighting for the propensity scores of each missingness indicator in the graph, the estimation of the causal effect
appears to be unbiased but has wider variance.
"""

import networkx as nx
import pandas as pd
import numpy as np
from scipy.special import expit
from listMAdj import *
from adjustment import *

def createTestGraph():
    """
    This test graph has no valid m-adjustment set, but its causal effect is recoverable via the IPW method.
    """
    G = nx.DiGraph()
    nodes = [('C', 'R_C'),('A', 'R_A'),('Y', 'R_Y')]
    G.add_nodes_from(['C','R_C','A','R_A','Y','R_Y'])

    G.add_edges_from([('C','A'),('C','Y'),('C','R_Y'),('C','R_A'),('A','Y'),('A','R_C'),('A','R_Y'),('Y','R_C'),('Y','R_A')])

    return (G, nodes)

def dataTesting():
    np.random.seed(0)

    # data generation process
    size = 2000
    C = np.random.normal(0, 1, size)
    A = np.random.binomial(1, expit(C), size)
    #print(A.sum())
    Y = 1.5 + 2.5*C + 2*A + np.random.normal(0, 1, size)
    data = pd.DataFrame({"Y": Y, "A": A, "C": C})

    print('fully observed data:', backdoor_adjustment('Y', 'A', ['C'], data), compute_confidence_intervals('Y', 'A', ['C'], data, "backdoor"))

    # create missingness mechanisms
    R_C = np.random.binomial(1, expit(Y+A-0.3), size)
    R_A = np.random.binomial(1, expit(C+Y+0.1), size)
    R_Y = np.random.binomial(1, expit(A+C+0.7), size)
    # print(R_C.sum())
    # print(R_A.sum())
    # print(R_Y.sum())
    # make sure that at least 70% of the data is still observed
    assert R_C.sum() >= size*0.7, 'too many missing values in R_C'
    assert R_A.sum() >= size*0.7, 'too many missing values in R_A'
    assert R_Y.sum() >= size*0.7, 'too many missing values in R_Y'

    # create observed variables
    C_observed = C.copy()
    A_observed = A.copy()
    Y_observed = Y.copy()
    to_delete = []
    for i in range(size):
        if R_C[i] == 0:
            C_observed[i] = 99999
        if R_A[i] == 0:
            A_observed[i] = -1
        if R_Y[i] == 0:
            Y_observed[i] = 99999
        if R_C[i] == 0 or R_A[i] == 0 or R_Y[i] == 0:
            to_delete.append(i)

    data_missing = data.copy()
    data_missing["R_C"] = R_C
    data_missing["R_A"] = R_A
    data_missing["R_Y"] = R_Y
    data_missing["C_observed"] = C_observed
    data_missing["A_observed"] = A_observed
    data_missing["Y_observed"] = Y_observed

    data_missing = data_missing[data_missing["C_observed"] != 99999]
    data_missing = data_missing[data_missing["A_observed"] != -1]
    data_missing = data_missing[data_missing["Y_observed"] != 99999]
    # there are about 1000 rows of data remaining
    #print(data_missing)

    # Fit the model for the propensity score.
    # We know the exact formula for the propensity scores, so do we need to fit a model here? Trying to fit a model also gives
    # me a perfect separation error.
    # model_R_C = sm.GLM.from_formula(formula='R_C ~ A_observed + Y_observed', data=data_missing, family=sm.families.Binomial()).fit()
    # model_R_A = sm.GLM.from_formula(formula='R_A ~ C_observed + Y_observed', data=data_missing, family=sm.families.Binomial()).fit()
    # model_R_Y = sm.GLM.from_formula(formula='R_Y ~ A_observed + C_observed', data=data_missing, family=sm.families.Binomial()).fit()

    # remove unobserved rows of data from each of the observed variables
    C_observed = np.delete(C_observed, to_delete, None)
    #print(C_observed.size)
    A_observed = np.delete(A_observed, to_delete, None)
    #print(A_observed.size)
    Y_observed = np.delete(Y_observed, to_delete, None)
    #print(Y_observed.size)

    # We know the exact formula for the propensity scores, so we just replicate it. The probability of a missingness
    # indicator being 1 is represented by the following functions. The numpy arrays of observed variables contain only
    # rows where the data is observed.
    propensityScore_R_C = expit(Y_observed+A_observed-0.3)
    propensityScore_R_A = expit(C_observed+Y_observed+0.1)
    propensityScore_R_Y = expit(A_observed+C_observed+0.7)

    # add the weights to the dataframe
    data_missing["weights"] = 1/(propensityScore_R_C*propensityScore_R_A*propensityScore_R_Y)

    # sample the data with replacement according to the weights
    weighted_data_missing = data_missing.sample(n=len(data_missing), replace=True, weights="weights")
    
    print('partially observed data:', backdoor_adjustment('Y_observed', 'A_observed', ['C_observed'], data_missing), compute_confidence_intervals('Y_observed', 'A_observed', ['C_observed'], data_missing, "backdoor"))
    print('partially observed weighted data:', backdoor_adjustment('Y_observed', 'A_observed', ['C_observed'], weighted_data_missing), compute_confidence_intervals('Y_observed', 'A_observed', ['C_observed'], weighted_data_missing, "backdoor"))

if __name__ == "__main__":
    testGraph = createTestGraph()
    G = testGraph[0]
    nodes = testGraph[1]
    print('m-adjustment sets:', listMAdj(G, 'A', 'Y', nodes))
    dataTesting()
