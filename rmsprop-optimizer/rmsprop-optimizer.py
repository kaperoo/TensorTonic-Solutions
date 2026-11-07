import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    new_s = beta*np.asarray(s)+(1-beta)*np.asarray(g)**2
    new_w = np.asarray(w)-(lr/np.sqrt(new_s+eps))*np.asarray(g)
    return (new_w, new_s)