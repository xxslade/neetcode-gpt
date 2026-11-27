import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:

        n = len(y_true)
        z1 = np.matmul(W1, x) + b1
        y1Cap = np.maximum(0, z1)

        y2Cap = np.matmul(W2,y1Cap.T) + b2 
        Loss = np.mean((y_true-y2Cap)**2)

        print(y1Cap)
        
        diff2 = (y_true-y2Cap)
        y1CapReshaped = -y1Cap.reshape(1,len(y1Cap))
        diff2Reshaped = diff2.reshape(len(diff2), 1)


        flag = y1Cap > 0
        sm = np.sum(W2*(y_true-y2Cap), axis=0)
        sm = sm*flag
        smReshaped = sm.reshape(len(sm), 1)
        xReshaped = -np.array(x).reshape(1, len(x))

        print(np.matmul(smReshaped, xReshaped))
        dW1 = 2*np.matmul(smReshaped, xReshaped)/n
        dW2 = 2*np.matmul(diff2Reshaped, y1CapReshaped)/n
        db1 = -2*sm/n
        db2 = -2*diff2/n

        return {"loss":np.round(Loss, 4), "dW1":np.round(dW1, 4), "dW2":np.round(dW2, 4), "db1":np.round(db1,4), "db2":np.round(db2,4)}
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        pass
