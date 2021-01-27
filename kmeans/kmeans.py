from sklearn.datasets import make_blobs
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
import numpy as np
import time
### TO DO:
# fix Kplus
# bisecting
# testing

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y_true: original labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1 
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # ind is a tuple of 2 arrays, ([rows], [columns]) that grant optimal assignment in terms of linear assignment problem
    ind = linear_assignment(w.max() - w)
    # sum(np.diagonal(w)) * 1.0 / y_pred.size
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size


class K_Means:
    def __init__(self, k, tolerance = 0.01, max_iter = 100, k_plus=False):
        self.k = k
        self.tolerance = tolerance
        self.k_plus = k_plus
        self.max_iter = max_iter
    
    def __init_centroids(self):
        centroids = {}
        sample = self.data[(np.random.choice(self.data.shape[0], self.k, replace=False))]
        for i in range(self.k):
            centroids[i] = sample[i]
        return centroids
    
    def __init_centroids_plus(self):
        centroids = {}
        rows = self.data.shape[0]
        sample = self.data.copy()[(np.random.choice(rows, rows // 2, replace=False))]
        centroids[0] = sample[np.random.randint(rows // 2)]
        for k in range(1, self.k):
            dist = dict([[i, []] for i in range(len(centroids))])
            for j in range(sample.shape[0]):
                point = sample[j]
                for i in range(len(centroids)):
                    dist[i].append(np.linalg.norm(point - centroids[i]))
            if len(centroids) == 1:
                idx = max(dist, key=dist.get)
                centroids[k] = sample[idx]
            else:
                mtx = np.zeros((len(dist[0]), len(dist.keys())))
                for j in range(mtx.shape[1]):
                    mtx[:, j] = dist[j]
                dist_sum = []
                for j in range(mtx.shape[0]):
                    dist_sum.append(np.sum(mtx[j, :]))
                idx = dist_sum.index(max(dist_sum)) 
                centroids[k] = sample[idx]
                
        return centroids

    def __handle_empty_clusters(self, clusters, SSE_table):
        # highest SSE cluster
        hsc = clusters[np.argmax(SSE_table)]
        for i in range(self.k):
            if len(clusters[i]) == 0:
                self.centroids[i] = hsc[np.random.randint(len(hsc))]
                return

    def __assign_to_centroids(self):
        while(True):
            print('xd')
            clusters = dict([[i, []] for i in range(self.k)])
            SSE_table = np.zeros(self.k)
            for point in self.data:
                dist = []
                for i in range(self.k):
                    d = np.linalg.norm(point - self.centroids[i]) 
                    dist.append(d)
                    SSE_table[i] += np.square(d)
                clusters[dist.index(min(dist))].append(point)   
            if [] in clusters.values():
                self.__handle_empty_clusters(clusters, SSE_table)
                print('!!!')
            else:
                return clusters
        
    def __recalculate_centroids(self):
        prev = {}
        for i in range(self.k):
            new_centroid = np.average(self.clusters[i], axis=0)
            prev[i] = self.centroids[i]
            self.centroids[i] = new_centroid
        return prev
    
    def __is_fitting(self, prev):
        for i in range(self.k):
            ratio = np.sum((self.centroids[i] - prev[i]) / prev[i])
            if ratio > self.tolerance:
                return False
        return True

    def fit(self, data):
        self.data = data
        self.clusters = {}
        if self.k_plus: 
            self.centroids = self.__init_centroids_plus() 
        else: 
            self.centroids = self.__init_centroids()
        
        for i in range(self.max_iter):
            self.clusters = self.__assign_to_centroids()
            if self.__is_fitting(self.__recalculate_centroids()): 
                break
        
    def predict(self, data):
        clusters = []
        for d in data: 
            for cl in self.clusters.keys():
                for x in self.clusters[cl]:
                    if (d == x).all():
                        clusters.append(cl)
        return np.array(clusters)

if __name__ == '__main__':
    outcomes = []
    for i in range(1):
        score_plus = 0
        score_normal = 0
        score_km = 0
        t1 = time.time()
        for j in range(50):
            print(j)
            X, y_true = make_blobs(n_samples=600, centers=8,
                               cluster_std=0.60)
            k = K_Means(8, k_plus=True)
            k.fit(X)
            score_plus += np.sum(cluster_acc(y_true, k.predict(X)))
        t_plus = time.time() - t1
        t1 = time.time()
        for j in range(50):
            print(j)
            X, y_true = make_blobs(n_samples=600, centers=8,
                               cluster_std=0.60)
            km = KMeans(8)
            km.fit(X)
            score_km += np.sum(cluster_acc(y_true, km.predict(X)))
        t_km = time.time() - t1
        t1 = time.time()
        for j in range(50):
            print(j)
            X, y_true = make_blobs(n_samples=600, centers=8,
                               cluster_std=0.60)
            k = K_Means(8)
            k.fit(X)
            score_normal += np.sum(cluster_acc(y_true, k.predict(X)))
        t_normal = time.time() - t1
        outcomes.append(f"plus: {score_plus} t: {int(t_plus)} normal: {score_normal} t: {int(t_normal)} sklearn: {score_km} t: {int(t_km)}")
    print(outcomes)
