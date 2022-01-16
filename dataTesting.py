"""
This file contains sample graphs and sample datasets generated according to the conditional 
independencies implied by those graphs. It tests functions in the file listMAdj.py and bias
induced by missing data.

This file was written by Jacob Chen.
"""

import pandas as pd
import numpy as np
from scipy.special import expit
from listMAdj import *
from adjustment import *

def createTestGraph2():
    """
    test graph 2 has two valid m-adjustment sets
    """

    G = nx.DiGraph()
    nodes = [('X', None),('Y', None),('Z1', 'R_Z1'),('Z2', 'R_Z2')]
    G.add_nodes_from(['Z2','Z1','R_Z1','R_Z2','X','Y'])

    G.add_edges_from([('Z','R_Z1'),('Z2','R_Z2'),('Z2','X'),('Z1','X'),('Z1','Y'),('X','Y')])

    return (G, nodes)

def dataTesting2():
    np.random.seed(0)

    # can increase it to see how it behaves asymptotically
    # data generation process (DGP)
    size = 2000
    Z1 = np.random.normal(0, 1, size)
    Z2 = np.random.binomial(1, 0.5, size)
    X = np.random.binomial(1, expit(Z1 + Z1*Z2 + Z1**2), size)
    Y = 1.5 + 2.5*Z1 + 2*X + np.random.normal(0, 1, size)
    data = pd.DataFrame({"Y": Y, "X": X, "Z1": Z1, "Z2": Z2})

    # can also try adjusting Z1 and Z2 as well, expect higher variance but unbiased estimate
    print('fully observed data:', backdoor_adjustment('Y', 'X', ['Z1'], data), compute_confidence_intervals('Y', 'X', ['Z1'], data, "backdoor"))

    # produce a binary variables R_Z1 and R_Z2 that are both functions of Z2
    # a value of 1 indicates that the data is observed, and 0 indicates that the data is not observed
    R_Z1 = np.random.binomial(1, expit(Z2*5.5), size)
    R_Z2 = np.random.binomial(1, expit(Z2*5), size)
    # make sure that at least 70% of the data is still observed
    assert R_Z1.sum() >= size*0.7, 'too many missing values in R_Z1'
    assert R_Z2.sum() >= size*0.7, 'too many missing values in R_Z2'

    # create Z1observed and Z2observed
    # whenever R_Z1 or R_Z2 is observed, copy in actual value of Z1
    # whenever R_Z1 or R_Z2 is not observed, put in an invalid value
    Z1_observed = Z1.copy()
    Z2_observed = Z2.copy()
    for i in range(size):
        if R_Z1[i] == 0:
            Z1_observed[i] = 99999
        if R_Z2[i] == 0:
            # assign some number outside of range of normal binary variable
            Z2_observed[i] = -1

    # create a second data set augmented with observed values and missingness mechanisms
    data_missing = data.copy()
    data_missing["R_Z1"] = R_Z1
    data_missing["R_Z2"] = R_Z2
    data_missing["Z1_observed"] = Z1_observed
    data_missing["Z2_observed"] = Z2_observed

    # drop the rows where values are missing
    data_missing = data_missing[data_missing["Z1_observed"] != 99999]
    data_missing = data_missing[data_missing["Z2_observed"] != -1]

    # when calculating backdoor_adjustment, we only need to pass in Z1, we don't need R_Z1 since conditioning is implied
    print('partially observed data:', backdoor_adjustment('Y', 'X', ['Z1_observed'], data_missing), compute_confidence_intervals('Y', 'X', ['Z1_observed'], data_missing, "backdoor"))

def createTestGraph3():
    """
    test graph 3 should not fulfill the m_adjustment criterion since we are forced to condition on the
    missingness mechanism of Z1, which is the descendant of a collider
    if there were no missing data, Z1 is a sufficient adjustment set
    """

    G = nx.DiGraph()
    nodes = [('X', None),('Y', None),('Z1', 'R_Z1'),('Z2', None)]
    G.add_nodes_from(['X','Y','Z1','R_Z1','Z2'])

    G.add_edges_from([('X','Y'),('Z1','X'),('Z1','Y'),('X','Z2'),('Y','Z2'),('Z2','R_Z1')])

    return (G, nodes)

def dataTesting3():
    # this function generates a dataset according to graph 3 and tests for its causal effect
    np.random.seed(0)

    # data generation
    size = 2000
    Z1 = np.random.normal(0, 1, size)
    X = np.random.binomial(1, expit(Z1 + Z1**2), size)
    Y = 1.5 + 2.5*Z1 + 2*X + np.random.normal(0, 1, size)
    Z2 = 4 + 3*X + Y**2 + np.random.normal(0, 1, size)
    data = pd.DataFrame({"Y": Y, "X": X, "Z1": Z1, "Z2": Z2})
    print('fully observed data:', backdoor_adjustment('Y', 'X', ['Z1'], data), compute_confidence_intervals('Y', 'X', ['Z1'], data, "backdoor"))

    # produce a binary variables R_Z1 that is a function of Z2
    # a value of 1 indicates that the data is observed, and 0 indicates that the data is not observed
    R_Z1 = np.random.binomial(1, expit(3*Z2-22), size)
    assert R_Z1.sum() >= size*0.7, 'too many missing values in R_Z1'

    # create Z1_observed
    Z1_observed = Z1.copy()
    for i in range(size):
        if R_Z1[i] == 0:
            Z1_observed[i] = 99999

    # create a second data set augmented with observed values and missingness mechanisms
    data_missing = data.copy()
    data_missing["R_Z1"] = R_Z1
    data_missing["Z1_observed"] = Z1_observed

    # drop the rows where Z1 is missing
    data_missing = data_missing[data_missing["Z1_observed"] != 99999]

    print('partially observed data:', backdoor_adjustment('Y', 'X', ['Z1_observed'], data_missing), compute_confidence_intervals('Y', 'X', ['Z1_observed'], data_missing, "backdoor"))

def createTestGraph4():
    """
    test graph 4 should not fulfill the m_adjustment criterion since it is impossible to
    d-separate Y and its missingness mechanism no matter who is the ancestor of R_Y
    """
    G = nx.DiGraph()
    nodes = [('X', None),('Y', 'R_Y'),('Z1', None),('Z2', None),('Z3', None)]
    G.add_nodes_from(['X','Y','R_Y','Z1','Z2','Z3'])

    G.add_edges_from([('X','Y'),('Z1','X'),('Z1','Y'),('Z2','Z1'),('Z2','Z3'),('Z3','Y'),('Z3','R_Y')])

    return (G, nodes)

def dataTesting4():
    # this function tests data for graph 4
    np.random.seed(0)

    # data generation
    size = 2000
    Z2 = np.random.normal(0, 1, size)
    Z1 = 3.75*Z2 + np.random.normal(0, 1, size)
    Z3 = 2.34*Z2 + np.random.normal(0, 1, size)
    X = np.random.binomial(1, expit(Z1), size)
    Y = 1.5 + 2.5*Z1 + 2.22*Z3 + 2*X + np.random.normal(0, 1, size)
    data = pd.DataFrame({"Y": Y, "X": X, "Z1": Z1, "Z2": Z2, "Z3": Z3})
    print('fully observed data:', backdoor_adjustment('Y', 'X', ['Z1','Z3'], data), compute_confidence_intervals('Y', 'X', ['Z1','Z3'], data, "backdoor"))

    # produce a binary variables R_Y that is a function of Z3
    R_Y = np.random.binomial(1, expit(3*Z3+4.2), size)
    print(R_Y.sum())
    assert R_Y.sum() >= size*0.7, 'too many missing values in R_Y'

    # create Y_observed
    Y_observed = Y.copy()
    for i in range(size):
        if R_Y[i] == 0:
            Y_observed[i] = 99999

    # create a second data set augmented with observed values and missingness mechanisms
    data_missing = data.copy()
    data_missing["R_Y"] = R_Y
    data_missing["Y_observed"] = Y_observed

    # drop the rows where Z1 is missing
    data_missing = data_missing[data_missing["Y_observed"] != 99999]
    print(data_missing)

    # when calculating backdoor_adjustment, we only need to pass in Z1 (don't need R_Z1 since conditioning is implied)
    print('partially observed data:', backdoor_adjustment('Y_observed', 'X', ['Z1', 'Z3'], data_missing), compute_confidence_intervals('Y_observed', 'X', ['Z1', 'Z3'], data_missing, "backdoor"))

def createAIDSGraph():
    """
    this graph uses the example of identifying the causal effect of condom use on diagnosis 
    of AIDS to test if missing data creates biased estimates of the causal effect

    If the missing data is a function of observed variables, then the causal effect should
    be biased. On the other hand, if data is missing completely at random (MCAR), the causal
    effect should not be biased.
    """
    G = nx.DiGraph()
    nodes = [('Age',None),('Partners','R_Partners'),('Income',None),('Drug','R_Drug'),('Condom','R_Condom'),('AIDS',None)]

    G.add_nodes_from(['Age','Partners','Income','Drug','Condom','AIDS','R_Partners','R_Drug','R_Condom'])

    G.add_edges_from([('Age','Partners'),('Age','Income'),('Age','Drug'),('Age','Condom'),('Partners','AIDS'),('Income','AIDS'),('Income','Drug'),
                      ('Income','Condom'),('Drug','AIDS'),('Drug','Condom'),('Condom','AIDS'),('Partners','R_Partners'),('Drug','R_Drug'),
                      ('Condom','R_Condom'),('Age','R_Condom')])
    return (G, nodes)

def dataTestAIDSGraph():
    np.random.seed(0)

    # data generation
    # is there data easily available? can check out data from final project also
    size = 2000
    Age = np.random.normal(30, 10, size)
    Partners = np.random.binomial(1, expit(-(0.1*Age) + 3))
    Income = Age*500 + np.random.normal(50000, 15000, size)
    Drug = np.random.binomial(1, expit(-(0.1*Age) + 1))
    Condom = np.random.binomial(1, expit(0.025*Age))
    AIDS = np.random.binomial(1, expit(-0.00001*Income + Partners + Drug + Condom - 5))
    data = pd.DataFrame({"Age": Age, "Partners": Partners, "Income": Income, "Drug": Drug, "Condom": Condom, "AIDS": AIDS})

    # outcome here is binary, should make sure we fit logistic regression model when outcome is binary
    print('fully observed data:', backdoor_adjustment('Condom', 'AIDS', ['Partners','Income','Drug'], data), compute_confidence_intervals('Condom', 'AIDS', ['Partners','Income','Drug'], data, "backdoor"))

    # produce missingness mechanisms that are functions of other variables in the graph
    R_Partners = np.random.binomial(1, expit(Partners+1), size)
    R_Drug = np.random.binomial(1, expit(Drug+1.3), size)
    R_Condom = np.random.binomial(1, expit(Condom+0.03*Age), size)
    # print(R_Partners.sum())
    # print(R_Drug.sum())
    # print(R_Condom.sum())
    assert R_Partners.sum() >= size*0.8
    assert R_Drug.sum() >= size*0.8
    assert R_Condom.sum() >= size*0.8
    amountMissing_R_Partners = R_Partners.sum()
    amountMissing_R_Drug = R_Drug.sum()
    amountMissing_R_Condom = R_Condom.sum()

    # create observed variables
    Partners_observed = Partners.copy()
    Drug_observed = Drug.copy()
    Condom_observed = Condom.copy()
    for i in range(size):
        if R_Partners[i] == 0:
            Partners_observed[i] = -1
        if R_Drug[i] == 0:
            Drug_observed[i] = -1
        if R_Condom[i] == 0:
            Condom_observed[i] = -1

    # create a second data set augmented with observed values and missingness mechanisms
    data_missing = data.copy()
    data_missing["R_Partners"] = R_Partners
    data_missing["Partners_observed"] = Partners_observed
    data_missing["R_Drug"] = R_Drug
    data_missing["Drug_observed"] = Drug_observed
    data_missing["R_Condom"] = R_Condom
    data_missing["Condom_observed"] = Condom_observed

    # drop the rows where data is missing
    data_missing = data_missing[data_missing["Partners_observed"] != -1]
    data_missing = data_missing[data_missing["Drug_observed"] != -1]
    data_missing = data_missing[data_missing["Condom_observed"] != -1]
    # print(data_missing.shape)

    print('partially observed data:', backdoor_adjustment_binary('Condom_observed', 'AIDS', ['Partners_observed','Income','Drug_observed'], data_missing), compute_confidence_intervals('Condom_observed', 'AIDS', ['Partners_observed','Income','Drug_observed'], data_missing, "backdoor_binary"))

    # create missingness mechanisms where values are missing completely at random
    R_Partners_random = np.random.binomial(1, amountMissing_R_Partners/size, size)
    R_Drug_random = np.random.binomial(1, amountMissing_R_Drug/size, size)
    R_Condom_random = np.random.binomial(1, amountMissing_R_Condom/size, size)
    assert R_Partners_random.sum() >= size*0.8
    assert R_Drug_random.sum() >= size*0.8
    assert R_Condom_random.sum() >= size*0.8

    # create observed variables
    Partners_observed_random = Partners.copy()
    Drug_observed_random = Drug.copy()
    Condom_observed_random = Condom.copy()
    for i in range(size):
        if R_Partners_random[i] == 0:
            Partners_observed_random[i] = -1
        if R_Drug[i] == 0:
            Drug_observed_random[i] = -1
        if R_Condom_random[i] == 0:
            Condom_observed_random[i] = -1

    data_missing_random = data.copy()
    data_missing_random["R_Partners_random"] = R_Partners_random
    data_missing_random["Partners_observed_random"] = Partners_observed_random
    data_missing_random["R_Drug_random"] = R_Drug_random
    data_missing_random["Drug_observed_random"] = Drug_observed_random
    data_missing_random["R_Condom_random"] = R_Condom_random
    data_missing_random["Condom_observed_random"] = Condom_observed_random

    # drop the rows where data is missing
    data_missing_random = data_missing_random[data_missing_random["Partners_observed_random"] != -1]
    data_missing_random = data_missing_random[data_missing_random["Drug_observed_random"] != -1]
    data_missing_random = data_missing_random[data_missing_random["Condom_observed_random"] != -1]
    # print(data_missing_random.shape)

    # make sure to use logistic regression technique here
    print('partially observed data at random:', backdoor_adjustment_binary('Condom_observed_random', 'AIDS', ['Partners_observed_random','Income','Drug_observed_random'], data_missing_random), compute_confidence_intervals('Condom_observed_random', 'AIDS', ['Partners_observed_random','Income','Drug_observed_random'], data_missing_random, "backdoor_binary"))
    

def createAIDSGraphV2():
    """
    this graph is a variant of the graph above except all missingness mechanisms are independent
    of observed and partially observed variables, this implies that valid m-adjustment sets should
    be the same as normal adjustment sets
    """

    G = nx.DiGraph()
    nodes = [('Age',None),('Partners','R_Partners'),('Income',None),('Drug','R_Drug'),('Condom','R_Condom'),('AIDS',None)]

    G.add_nodes_from(['Age','Partners','Income','Drug','Condom','AIDS','R_Partners','R_Drug','R_Condom'])

    G.add_edges_from([('Age','Partners'),('Age','Income'),('Age','Drug'),('Age','Condom'),('Partners','AIDS'),('Income','AIDS'),('Income','Drug'),
                      ('Income','Condom'),('Drug','AIDS'),('Drug','Condom'),('Condom','AIDS')])
    return (G, nodes)


if __name__ == "__main__":
    testGraph = createTestGraph2()
    G = testGraph[0]
    nodes = testGraph[1]
    print(listMAdj(G, 'X', 'Y', nodes))
    print('testing graph 2, has valid m-adjustment set')
    dataTesting2()
    print()

    testGraph = createTestGraph3()
    G = testGraph[0]
    nodes = testGraph[1]
    print(listMAdj(G, 'X', 'Y', nodes))
    print('testing graph 3, does not have valid m-adjustment set')
    dataTesting3()
    print()

    # test graph 4 does not seem to have biased results, perhaps effect of Z3 on R_Y is not 
    # strong enough?
    # the estimate of the causal effect when there is no missing data also seems to be biased
    testGraph = createTestGraph4()
    G = testGraph[0]
    nodes = testGraph[1]
    print(listMAdj(G, 'X', 'Y', nodes))
    print('testing graph 4, does not have valid m-adjustment set')
    dataTesting4()
    print()

    # MNAR dataset seems to be pretty unbiased, but MCAR dataset is pretty biased?
    testGraph = createAIDSGraph()
    G = testGraph[0]
    nodes = testGraph[1]
    print(listMAdj(G, 'Condom', 'AIDS', nodes))
    print('testing AIDS graph, first test is a version of graph where there is no valid m-adjustment set')
    print('second test uses assumption that missingness is caused at random, so m-adjustment set is same as normal adjustment set (data is MCAR)')
    dataTestAIDSGraph()

    # testGraph = createAIDSGraphV2()
    # G = testGraph[0]
    # nodes = testGraph[1]
    # print(listMAdj(G, 'Condom', 'AIDS', nodes))