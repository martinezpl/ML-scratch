import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=1):
        self.k = k
        self.p = 2

    def _minkowski_dist(self, v1, v2):
        return sum(abs(e1 - e2) ** self.p for e1, e2 in zip(v1, v2)) ** (1 / self.p)

    def fit(self, X, labels, p=2):
        self.p = p
        self.X = X
        self.Y = labels
        return self
    
    def predict(self, query, method='mean'):
        assert method in ['mean', 'mode']
        dist_label = []
        for i in range(len(self.X)):
           dist_label.append((self._minkowski_dist(query, self.X[i]), self.Y[i]))

        neighbours = [x[1] for x in sorted(dist_label, key = lambda x: x[0])[:self.k]]
            
        if method=='mean':
            return np.average(neighbours, axis=0)
        elif method=='mode':
            count = Counter(neighbours)
            return max(count, key=count.get)


