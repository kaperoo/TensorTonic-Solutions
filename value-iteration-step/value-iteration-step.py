import numpy as np
def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    ts = np.asarray(transitions)
    vs = np.asarray(values)
    rs = np.asarray(rewards)
    return list(np.max(rs+gamma*np.sum(ts*vs, axis=2), axis=1))
    