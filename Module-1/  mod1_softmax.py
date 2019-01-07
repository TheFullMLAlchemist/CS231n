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

# handling numeric stability of softmax function

f = np.array([123, 463, 789])

"""
p = np.exp(f) / np.sum(np.exp(f)) 

This is bad as the numbers are very large and might just blow up

Therefore we multiply a constant C on both numerator and denominator
s.t. the numbers does not blow up.
The most common value of C is: C = -max(f)

"""
f = -np.max(f)
p = np.exp(f) / np.sum(np.exp(f)) # This is safe to use as the numbers are very small



