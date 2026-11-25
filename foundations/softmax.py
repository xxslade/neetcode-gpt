import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        mx = np.max(z)
        z = z - mx*np.ones(len(z))
        expArr = np.exp(z)
        sumVal = np.sum(expArr)
        
        res = np.round(expArr/sumVal, 4)

        return res
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        pass
