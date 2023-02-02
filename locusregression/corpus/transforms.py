
import numpy as np

class ConvolutionalKernelTransformer:

    def __init__(self, window_size = 5):
        self.window_size = window_size

    def _add_lag(X, j):

        if j == 0:
            return X

        elif j > 0:

            return np.hstack([
                X[:,j:],
                np.repeat(X[:,-1], j, axis = -1)
            ])

        else:

            return np.hstack([
                np.repeat(X[:,0], -j, axis = -1),
                X[:,:j],
            ])


    def transform(self, X):

        overlap = (self.window_size - 1)/2

        assert isinstance(overlap, int)

        return np.vstack([
            self._add_lag(X.copy(), j) for j in range(-overlap, overlap + 1)
        ])