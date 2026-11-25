import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    T = np.asarray(T)
    points = np.reshape(np.asarray(points),(-1,3))
    ones = np.ones(len(points))
    points = np.column_stack((points,ones))
    
    p = T@points.T
    return p.T[:,:3][0] if len(ones)==1 else p.T[:,:3]
    
