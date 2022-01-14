# This file is a backup of my implementation for findProperCausalPaths() that uses a recursive
# implementation. I opted for an iterative implementation because I prefer the iterative implementation
# of DFS.

def findProperCausalPath(G, X, Y):
    """
    This function finds all proper causal paths from X to Y. A proper causal path is a directed path
    from X to Y that does not intersect with X except at the beginning of the path (we don't need 
    to worry about intersecting with X again, though, since G is acyclic).

    This function returns the causal path as a list of paths. Each path is represented by a list of 
    edges, with each edge being represented by a tuple.
    """
    # DFS from X to find proper causal paths from X to Y.
    startVertex = X

    paths = findProperCausalPathHelper(G, X, Y, startVertex, [], [])
    if len(paths) > 0:
        return paths
    else:
        print('no proper causal path found')
        return None
    
def findProperCausalPathHelper(G, X, Y, curVertex, curPath, paths):
    """
    Helper function for findProperCausalPath that implements a DFS for a directed acyclic graph.
    """
    # we've successfully found a proper causal path
    if curVertex == Y:
        return curPath

    for child in G.successors(curVertex):
        curPath.append((curVertex, child))
        # see if this path eventually takes us to Y
        result = findProperCausalPathHelper(G, X, Y, child, curPath.copy(), paths)
        # result can either be an empty list, or a list with a sequence of edges to Y
        if len(result) > 0:
            paths.append(result)
        # already tried going through child to go to Y, so remove it from curPath
        curPath.pop()

    # we are only interested in paths that start specifically from X
    if curVertex == X:
        return paths
    else:
        return []