from copy import copy
import numpy as np

class Perceptron:
    def __init__(self, learning_rate, max_iter):
        self.eta = learning_rate
        self.max_iter = max_iter
        self.weights = np.array([])
        self.bias = 0
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        
    def fit(self, X, y):
        weights = np.random.random(len(X[0]))
        bias = np.random.rand(1)
        it = 1
        while it < self.max_iter:
            for i in range(len(X)):
                z = np.dot(X[i], weights) + bias
                output = self.sigmoid(z)
                if not -0.5 < y[i] - output < 0.5:
                    weights += self.eta*(y[i] - z)*X[i]
                    bias += self.eta*(y[i] - z)
            if self.weights.all() == weights.all() and self.bias == bias:
                print(f"Convergence achieved after {it} iterations.")
                break
            self.weights, self.bias = copy(weights), copy(bias)
            it += 1
        return self
    
    def predict(self, X):
        return np.round(self.sigmoid(np.dot(X, self.weights) + self.bias))

if __name__ == '__main__':
    from sklearn.preprocessing import MinMaxScaler
    np.random.shuffle(training_data)
    model = Perceptron(0.01, 1000)
    scaler = MinMaxScaler()
    training_data_s = scaler.fit_transform(training_data[:, :2])
    test_data_s = scaler.transform(test_data[:, :2])
    model = model.fit(training_data_s, training_data[:, 2])

    ## model_evaluation:
    test = test_data_s[:, :2]
    gt = test_data[:, 2]
    points = 0
    for sample, truth in zip(test, gt):
        if truth == model.predict(sample):
            points += 1

    print("Model accuracy:", points/len(gt))
