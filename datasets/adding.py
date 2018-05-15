import numpy as np


class Adding:
    def __init__(self, T=200, n_train=50000, n_test=1000):
        self.T = T
        self.n_train = n_train
        self.n_test = n_test

        X_train, y_train = self.load_data(n_train)
        X_val, y_val = self.load_data(n_test)

        self.train = {'X': {'data': X_train, 'axes': ('N', 'F', 'REC')}, 'y': {'data': y_train, 'axes': ('N', 'Fo')}}

        self.test = {'X': {'data': X_val, 'axes': ('N', 'F', 'REC')}, 'y': {'data': y_val, 'axes': ('N', 'Fo')}}

    def load_data(self, N):
        """
        Args:
            N: # of data in the set
        """
        X_num = np.random.rand(N, 1, self.T)
        X_mask = np.zeros((N, 1, self.T))
        y = np.zeros((N, 1))
        for i in range(N):
            positions = np.random.choice(self.T, size=2, replace=False)
            X_mask[i, 0, positions[0]] = 1
            X_mask[i, 0, positions[1]] = 1
            y[i, 0] = X_num[i, 0, positions[0]] + X_num[i, 0, positions[1]]
        X = np.concatenate((X_num, X_mask), axis=1)
        return X, y