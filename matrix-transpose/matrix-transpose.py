import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    i = len(A)
    j = len(A[0])

    arr = np.ndarray((j,i))

    for k in range(i):
        for l in range(j):
            arr[l,k]=A[k][l]

    return arr