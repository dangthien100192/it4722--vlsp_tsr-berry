from typing import List
import numpy as np

def l2_normalize(vec: List[float]) -> List[float]:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()

def zero_vec(dim: int) -> List[float]:
    return [0.0] * dim
