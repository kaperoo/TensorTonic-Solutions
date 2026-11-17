import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if len(seqs)==0:
        return np.ndarray((0,0),int)

    N = len(seqs)
    if max_len:
        L = max_len
    else:
        L = 0
        for s in seqs:
            if len(s)>L:
                L=len(s)

    opt = np.full((N,L),pad_value)
    for i in range(len(opt)):
        l = min(L,len(seqs[i]))
        opt[i][:l] = np.asarray(seqs[i])[:l]

    return opt
        