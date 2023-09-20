import numpy as np
import statistics


class KNN:
    k_neighbors = 1
    X = None
    y = None

    def __init__(self, k_neighbors):
        self.k_neighbors = k_neighbors

    def fit(self, X, y):
        # Fitting model data into local variable
        self.X = X
        self.y = y

    def predict(self, X_input):
        # Predicting the value of X_input
        res = []
        for x_i in X_input:
            neighbors = self.calculate(x_i)
            majority = []
            for n_i in neighbors:
                majority.append(n_i[1])

            res.append(statistics.mode(majority))

        return res

    def calculate(self, X_input):
        # Calculating distance of euclidean using np
        # Will return the k number of neighbors data point
        distance = []
        for i, x_i in enumerate(self.X):
            d = np.sqrt(np.square(X_input - x_i))
            distance.append([d[0], self.y[i]])

        distance.sort()
        return distance[0:self.k_neighbors]
