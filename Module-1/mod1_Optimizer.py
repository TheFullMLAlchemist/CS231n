import numpy as np 
import pandas as pd 
import matplotlib as plt 

# This is not an exact function but reflects the functioning of load_CIFAR10
def load_CIFAR10(address):
    data = pd.read_csv(address)
    Y = data["tag"]
    X = data.drop("tag", axis=0)
    Xtr = X[:50000]
    Ytr = Y[:50000]
    Xte = X[50000:]
    Yte = Y[50000:]
    return Xtr, Ytr, Xte, Yte

# let the address of the CIFAR10 be 'data/cifar10/'

Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/')

# flatten out all images to be 1D
# Each image is 32x32 pixels and 3 for RGB

Xtr_row = Xtr.reshape(Xtr.shape[0], 32*32*3) 
Xte_row = Xte.reshape(Xte.shape[0], 32*32*3)

# Shape of Xtr_row = (50000, 3072)
# Shape of Xte_row = (10000, 3072)
# Shape of Ytr = (1, 50000) containing numbers 0-9 for 10 different classes
# Shape of Yte = (1, 10000) with values 0-9 for 10 different classes

# Loss function
def L(X, y, W):
    delta = 1.0
    scores = W.dot(X)
    margins = np.maximum(0, scores - scores[y]+delta)
    loss = np.sum(margins)
    return loss


# Optimization 
X_train = Xtr_row
Y_train = Ytr
X_test = Xte_row
Y_test = Yte

# Strategy-1 : Random Search
bestloss = float("inf") # Assign bestloss default to infinite
for num in xrange(1000):
    W = np.random.randn(10, 3073) * .0001 # Generate a random number
    loss = L(X_train, Y_train)
    if loss < bestloss:
        bestloss = loss
        bestW = W 
    print("in attemt %d the loss was %f, best %f" % (num, loss, bestloss))

scores = Wbest.dot(X_test) # Calculates the score
Y_test_predict = np.argmax(scores, axis=0) # Finds the index with maximum score
np.mean(Y_test_predict == Y_test) # Calculates the accuracy 

# 0.1555 was observed 

# Strategy-2: Random Local Search

W = np.random.randn(10, 3073) * 0.001 # Generate random starting W
bestloss = float("inf")
for i in range(1000):
    step_size = .0001
    Wtry = W + np.random.randn(10, 3073) * step_size
    loss = L(X_train, Y_train)
    if loss < bestloss:
        

