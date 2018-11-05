import numpy as np
import time


def softmax(x, d=False):
    if d:
        func = softmax(x)
        return func * (1 - func)
    else:
        res = np.zeros(x.shape)
        for i,row in enumerate(x):
            res[i] = np.exp(row)
            s = res[i].sum()
            res[i] = res[i] / s
        return res


def crossentropy(y, u, d=False):
    if d:
        return (y + 0.0001) / (u + 0.0001);
    else:
        pass


def euclid_error(y, y1, d=False):
    if d:
        return y - y1
    else:
        return 1/2 * sum((y - y1) ** 2, axis=0)


def logistic(x, d=False):
    if d:
        func = logistic(x)
        return func * (1 - func)
    return 1 / (1 + np.exp(-x))


def mix(x, y):
    t = np.concatenate((x, y), axis=1)
    np.random.shuffle(t)
    x = t[:, :x.shape[1]]
    y = t[:, x.shape[1]:]
    return x, y


class Classifier:
    def __init__(self, hidden_neurons=(), eps=0.001, max_iter=1,
                 batch_size=100, num_epochs=10, learn_rate=1):
        self.hidden_neurons = hidden_neurons
        self.hidden_layers = len(self.hidden_neurons)
        self.activation = logistic
        self.error = euclid_error
        self.hidden_neurons = hidden_neurons
        self.eps = eps
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learn_rate = learn_rate

    np.random.seed(0)

    def fit_batch(self, x, y):
        for j in range(self.max_iter):
            self.layers[0] = x

            for i in range(1, len(self.layers) - 1):
                self.layers[i] = self.activation(np.dot(self.layers[i - 1], self.weights[i - 1]))
            self.layers[-1] = self.activation(np.dot(self.layers[-2], self.weights[-1]))

            err = self.error(y, self.layers[-1], d=True)
            if np.mean(np.abs(err)) < self.eps:
                return

            for i in range(len(self.layers) - 2, -1, -1):
                diff = err * self.activation(self.layers[i + 1], d=True)
                err = diff.dot(self.weights[i].T)
                self.weights[i] += self.learn_rate * self.layers[i].T.dot(diff)

    def predict(self, x):
        self.layers[0] = x
        for i in range(1, len(self.layers) - 1):
            self.layers[i] = self.activation(np.dot(self.layers[i - 1], self.weights[i - 1]))

        self.layers[-1] = self.activation(np.dot(self.layers[-2], self.weights[-1]))

        predictions = np.zeros(self.layers[-1].shape[0])
        for i, prs in enumerate(self.layers[-1]):
            max_pr = 0
            max_class = 0

            for cl, pr in enumerate(prs):
                if pr > max_pr:
                    max_pr = pr
                    max_class = cl
            predictions[i] = max_class
        return predictions

    def fit(self, x, y):
        N, M = x.shape
        initX = x
        initY = y

        countClass = 0
        for c in y:
            if countClass - 1 < c:
                countClass = c + 1
        yy = np.zeros((N, countClass))

        for i in range(N):
            for j in range(countClass):
                if y[i] == j:
                    yy[i][j] = 1
        y = yy
        K = y.shape[1]

        if N < self.batch_size:
            self.batch_size = N

        self.layers = []
        self.layers.append(np.ndarray((self.batch_size, M)))
        for k in self.hidden_neurons:
            self.layers.append(np.ndarray((self.batch_size, k)))
        self.layers.append(np.ndarray((self.batch_size, K)))

        self.weights = []
        for i in range(1, len(self.layers)):
            self.weights.append(2 * np.random.random((self.layers[i - 1].shape[1], self.layers[i].shape[1])) - 1)

        countBatch = int(N / self.batch_size)

        for j in range(self.num_epochs):
            print("Epoch №" + str(j + 1))
            start_time = time.clock()
            x, y = mix(x, y)

            for i in range(countBatch):
                self.fit_batch(x[i:i + self.batch_size], y[i:i + self.batch_size])

            print('\tPredict error: ' + str(self.count_error(initX, initY)))
            print('\tSpent time is ' + str(time.clock() - start_time))

    def count_error(self, x, y):
        error = 0
        pred_y = self.predict(x)

        for i in range(x.shape[0]):
            if pred_y[i] != y[i]:
                error += 1.0

        result = error / x.shape[0]
        return result
