import numpy as np
from source import getdata
import time
from source.Net import Classifier
import sys

epochs = int(sys.argv[1])
l_rate = float(sys.argv[2])

print("-- Input parameters --")
print("Epochs: " + str(epochs))
print("Learning rate is " + str(l_rate))

N_train = 60000
N_test = 10000

print("\n-- Initialize --")
t = time.clock()
train_images = getdata.train_images()[:N_train]
train_labels = getdata.train_labels()[:N_train]
test_images = getdata.test_images()[:N_test]
test_labels = getdata.test_labels()[:N_test]

X_train = np.zeros((N_train, 784)) # 784 = 28 * 28 from image sizes
for i, pic in enumerate(train_images):
    X_train[i] = pic.flatten()

X_test = np.zeros((N_test, 784))
for i, pic in enumerate(test_images):
    X_test[i] = pic.flatten()

Y_train = train_labels
Y_test = test_labels

print("\n-- Classifier --")
net = Classifier(hidden_neurons=(800, ), eps=0.001, max_iter=1,
    batch_size=100, num_epochs=epochs, learn_rate=l_rate)
net.fit(X_train, Y_train)

Y_train_pre = net.predict(X_train)
Y_test_pre = net.predict(X_test)

print("\n-- Errors --")
err_train = 0
for i in range(N_train):
    if Y_train_pre[i] != Y_train[i]:
        err_train += 1.0
err_train = err_train / N_train
print("Train accuracy is " + str(1-err_train))

err_test = 0
for i in range(N_test):
    if Y_test_pre[i] != Y_test[i]:
        err_test += 1.0
err_test = err_test / N_test
print("Test accuracy is " + str(1-err_test))

t = time.clock() - t
print(t)
