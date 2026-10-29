import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    dim = len(X[-1])
    # w = np.zeros((1,dim))
    w = np.zeros(dim)
    b = 0
    
    for i in range(steps):
        # print(X@w.T)
        # print(np.ndarray((1,len(X)),buffer=b))
        pred = _sigmoid(X@w.T+np.array([b]*len(X)))
        # print(pred)
        # print(y)
        
        grad_w = X.T @ (pred-y).T/len(y)
        grad_b = np.mean(pred-y)
        # print(grad_w)
        w = w - lr*grad_w
        # print(w)
        b = b - lr*grad_b

    return (w,b)
        