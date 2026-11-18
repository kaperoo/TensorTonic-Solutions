import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    pos = np.reshape(np.array([i for i in range(seq_len)]), (seq_len,1))
    freq = np.reshape(np.array([i for i in range(int(np.ceil(d_model/2)))]), (1,int(np.ceil(d_model/2))))

    w = pos/(base**(2*freq/d_model))
    res = np.ndarray((seq_len,d_model))
    w_i = 0
    for i in range(d_model):
        if i%2==0:
            res[:,i]=np.sin(w[:,w_i])
        else:
            res[:,i]=np.cos(w[:,w_i])
            w_i+=1

    return res