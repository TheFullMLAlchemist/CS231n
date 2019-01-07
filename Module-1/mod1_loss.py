import numpy as np 
import pandas as pd 


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


# unvectorized implementation of loss function
# x is a column vector representing an image with
# an appended bais dimension in the 3073-rd position
# (bias trick)
# y is an integer giving index of correct class
# W is the weight matrix (10x3073)

def L_i(x, y, W):
    delta = 1.0
    scores = W.dot(x) # shape of score 10x1
    correct_class_score = scores[y]
    D = W.shape[0] # number of classes, i.e. 10
    loss_i = 0.0
    for j in xrange(D): # iterate over all the wrong classes
        if j==y:
            # skip for the true class to only loop over incorrect classes
            continue
        loss_i += max(0, score[j]-correct_class_score+delta)
    return loss


# Half-Vectorized implementation of loss function
# x is a column vector representing an image with
# an appended bais dimension in the 3073-rd position
# (bias trick)
# y is an integer giving index of correct class
# W is the weight matrix (10x3073)

def L_i_half_vectorized(x, y, W):
    delta = 1.0
    scores = W.dot(x)
    # compute the margins for all classes in one vector opertion
    margins = np.maximum(0, scores-scores[y] + delta)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i


def L_i_vectorized(X, y, W):
    delta = 1.0
    scores = W.dot(X)
    margins = np.maximum(0, scores - scores[y]+delta)
    loss = np.sum(margins)
    return loss

# we took delta = 1.0 because it is found that it is safe to use in all cases
# magintude of W has a direct effect on loss.
# therefore we use regularization techniques 