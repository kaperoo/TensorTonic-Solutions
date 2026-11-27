import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    if len(y) == 0:
        return 0.0

    cls = set(y)

    y = np.asarray(y)

    s = 0
    for c in cls:
        p = np.count_nonzero(y==c)/len(y)
        s += p*np.log2(p)
    
    return -s