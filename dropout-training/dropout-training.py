import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    p_inv = 1/(1-p)
    x = np.asarray(x)
    rand = rng.random(x.shape)
    
    scales = np.vectorize(lambda i: 0 if i > 1-p else p_inv)(rand)
    return (scales*x, scales)


    
    