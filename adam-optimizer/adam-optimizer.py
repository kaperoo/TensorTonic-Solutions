import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    m_t = beta1*np.asarray(m)+(1-beta1)*np.asarray(grad)
    v_t = beta2*np.asarray(v)+(1-beta2)*(np.asarray(grad)**2)

    b_m_t = m_t/(1-beta1**t)
    b_v_t = v_t/(1-beta2**t)
    params = np.asarray(param) - lr*b_m_t/(np.sqrt(b_v_t)+eps)

    return(params, m_t, v_t)