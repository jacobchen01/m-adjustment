import networkx as nx
import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy.special import expit

def findProperCausalPath(G, X, Y):
    """
    This function finds all proper causal paths from X to Y. A proper causal path is a directed path
    from X to Y that does not intersect with X except at the beginning of the path (we don't need 
    to worry about intersecting with X again, though, since G is acyclic).

    This function returns the causal path as a list of paths. Each path is represented by a list of 
    edges, with each edge being represented by a tuple.
    """
    paths = []

    # DFS from X to find proper causal paths from X to Y.
    # pathSoFar is a stack that keeps track of the path we have built so far
    pathSoFar = []
    # stk is the DFS stack, which keeps track of which element to go to next
    # each element on the stack remembers its source vertex, which helps us back track
    stk = []
    # the treatment does not have a parent in the proper causal path
    stk.append((X, None))
    while len(stk) > 0:
        curVertex = stk.pop()
        # remove vertices in pathSoFar until we get to the vertex that put curVertex on the stack
        # while the element that put the last element of pathSoFar on the stack is different from curVertex
        while len(pathSoFar) > 0 and (pathSoFar[-1] != curVertex[1]):
            pathSoFar.pop()
        pathSoFar.append(curVertex[0])
        if curVertex[0] == Y:
            paths.append(pathSoFar.copy())

        for child in G.successors(curVertex[0]):
            stk.append((child, curVertex[0]))

    # convert the list paths into a new list in the format specified in the docstring
    pathEdges = []
    for path in paths:
        thisPath = []
        for i in range(0, len(path)-1):
            thisPath.append((path[i], path[i+1]))
        pathEdges.append(thisPath)
        
    return pathEdges

def createProperBackdoorGraph(G, paths):
    """
    This function returns a networkx directed graph in the form of a proper backdoor graph, a
    graph where the first edge of every proper causal path from X to Y is removed.

    paths is a list containing all of the proper causal paths from X to Y.
    """
    G_copy = G.copy()
    for path in paths:
        firstEdge = path[0]
        source = firstEdge[0]
        sink = firstEdge[1]
        if G_copy.has_edge(source, sink):
            G_copy.remove_edge(source, sink)
    return G_copy

def createGXBarAbove(G, X):
    """
    This function returns a networkx directed graph where all incoming edges to X are deleted.
    """
    G_copy = G.copy()
    for parent in G.predecessors(X):
        G_copy.remove_edge(parent, X)
    return G_copy

def createGXBarBelow(G, X):
    """
    This function returns a networkx directed graph where all outgoing edges from X are deleted.
    """
    G_copy = G.copy()
    for child in G.successors(X):
        G_copy.remove_edge(X, child)
    return G_copy

def isAncestor(G, X, V):
    """
    This function returns True if X is an ancestor of any variable in list V. It returns False otherwise.
    """
    for vertex in V:
        stk = []
        stk.append(vertex)
        while len(stk) > 0:
            curVertex = stk.pop()
            # X is an ancestor of one of the variables in V
            if curVertex == X:
                return True
            for parent in G.predecessors(curVertex):
                stk.append(parent)
    
    return False

def findDescendants(G, source):
    """
    This function finds all the descendants of vertex source in graph G and returns them in a list,
    including the source vertex itself.
    """
    descendants = []

    stk = []
    stk.append(source)
    while len(stk) > 0:
        curVertex = stk.pop()
        descendants.append(curVertex)
        for child in G.successors(curVertex):
            stk.append(child)

    return descendants

def listMAdj(G, X, Y, V):
    """
    G is the DAG augmented with missingness mechanisms.
    V is the set of all observed and unobserved variables, each element in V is a tuple, the first
    element of the tuple is the variable itself, the second element is the missingness indicator
    for the variable if it is partially observed and None otherwise
    X is the treatment, Y is the outcome. Both X and Y are in the format outlined above as well.
    
    This function iterates over all possible combinations of adjustment sets and tests each set to
    see if it fulfills the M-adjustment criterion.

    This function returns a tuple. The first element is a list of all valid M-adjustment sets. 
    The second element is the minimum adjustment set with the smallest number of variables.
    """
    # find D_pcp, the set of descednants of those variables in a proper causal path from X to Y
    D_pcp = set()
    properCausalPaths = findProperCausalPath(G, X, Y)
    # for each possible causal path from X to Y
    for path in properCausalPaths:
        # for each edge in that path
        for edge in path:
            # for each vertex in an edge (there are only 2)
            for vertex in edge:
                # get all the descendants of this vertex
                descendants = findDescendants(G, vertex)
                # add the descendants to D_pcp
                for var in descendants:
                    D_pcp.add(var)
    # print('D_pcp', D_pcp)

    # There are a total of 2**varNum possible sets that could fulfill the M-adjustment criterion.
    varNum = len(V)
    validSets = []
    bestSet = None
    # Use a for loop that goes from 0 to 2**varNum. Convert the number to binary, and use it to 
    # represent the set Z we are currently testing for the M-adjustment criterion.
    for i in range(0, 2**varNum):
        # can also use iterative tools here instead of binary strings
        binString = bin(i)[2:].zfill(varNum)
        # print(binString)

        # Z is the adjustment set we are testing for.
        Z = []
        # R_W contains the missingness indicators of X, Y, and Z
        R_W = []
        for j in range(0, varNum):
            if binString[j] == '1':
                Z.append(V[j][0])
                if V[j][0] != X and V[j][0] != Y and V[j][1] != None:
                    R_W.append(V[j][1])
            if V[j][0] == X or V[j][0] == Y:
                if V[j][1] != None:
                    R_W.append(V[j][1])
        # print()
        # print(Z)
        # print(R_W)

        # flag indiciates whether Z fulfills condition 1 or not.
        flag = True
        # 1. No element of Z is on, or is the descendant of a variable on, the proper causal path from X to Y (D_pcp).
        for vertex in Z:
            if vertex in D_pcp:
                flag = False
                break
        if not flag:
            # condition 1 is not fulfilled
            #print('condition 1 failed')
            continue

        # 2. Y is d-separated from X given Z and R_W in the proper backdoor graph of G with respect to X and Y.
        G_pbd = createProperBackdoorGraph(G, properCausalPaths)
        if not nx.d_separated(G_pbd, {Y}, {X}, Z + R_W):
            # condition 2 is not fulfilled
            #print('condition 2 failed')
            continue

        # 3. Y and R_W are d-separated given X in G where all incoming edges from X are deleted.
        GXBarAbove = createGXBarAbove(G, X)
        if not nx.d_separated(GXBarAbove, {Y}, R_W, {X}):
            # condition 3 is not fulfilled
            #print('condition 3 failed')
            continue

        # 4. All X is either not an ancestor of R_W or is d-separated from Y in G where all outgoing edges 
        # from X are deleted.
        if isAncestor(G, X, R_W):
            GXBarBelow = createGXBarBelow(G, X)
            if not nx.d_separated(GXBarBelow, {X}, {Y}, {}):
                # condition 4 is not fulfilled
                #print('condition 4 failed')
                continue

        if bestSet == None or len(Z) < len(bestSet):
            bestSet = Z
        validSets.append(Z)
    
    return (validSets, bestSet)

def createTestGraph1():
    # test graph 1 tests for proper causal paths
    G = nx.DiGraph()
    nodes = [('U', None),('V', None),('A', None),('W', None),('X', None),('T', None),('C', None),('B', None),('Y', None),('Z', None)]
    G.add_nodes_from(['U','V','A','W','X','T','C','B','Y','Z'])
    G.add_edges_from([('U','W'),('U','A'),('V','W'),('V','X'),('V','T'),('A','C'),('A','B'),('A','Y'),('W','B'),('W','Y'),('X','Y'),('T','Z'),('B','Y')])

    return (G, nodes)

def createTestGraph():
    # this test graph is a simple graph that tests the output for proper causal paths
    G = nx.DiGraph()

    G.add_nodes_from(['A','M1','M2','Y','C1','C2','C3','C4','C5'])
    G.add_edges_from([('A', 'M1'),('A','M2'),('M1','Y'),('M2','Y'),('C1','C3'),('C1','C4'),('C2','C4'),('C2','C5'),
                      ('C3','A'),('C4','A'),('C4','M1'),('C4','Y'),('C5','Y'),('M1','M2')])
    # test graph after adding M1 -> M2 path

    return G

if __name__ == "__main__":
    G = createTestGraph()
    paths = findProperCausalPath(G, 'A', 'Y')
    assert len(paths) == 3 and ([('A','M1'),('M1','Y')] in paths) and ([('A','M2'),('M2','Y')] in paths) and ([('A','M1'),('M1','M2'),('M2','Y')]), 'wrong proper causal paths'

    G = createTestGraph1()[0]
    paths = findProperCausalPath(G, 'U', 'Y')
    assert len(paths) == 4 and ([('U','W'),('W','Y')] in paths) and ([('U','W'),('W','B'),('B','Y')] in paths) and ([('U','A'),('A','Y')] in paths) and ([('U','A'),('A','B'),('B','Y')] in paths), 'wrong proper causal paths'