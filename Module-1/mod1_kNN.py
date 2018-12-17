import numpy as np 
import pandas as pd 
from statistics import mode
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

# Define a Nearest Neighbor classifier class
class NearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self, X, y):
        # X is N x D where each row is an example. Y is 1D of size N
        self.Xtr = X
        self.ytr = y

    def predict(self, X, k):
        # X is N x D where each row is an example we wish to predict label for
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        predicted_list = []

        # loop over all test rows
        for i in xrange(num_test):
            # find the nearest training image to the ith test image
            # using the L1 distance (sum of absolute value difference)
            distances = np.sum(np.abs(self.Xtr - X[i:]), axis=1)
            # Get the index with smallest distance
            for j in range(k):
                min_index = np.argmin(distances)
                distances.delete(min_index)
                predicted_list.append(min_index)

            # predict the label of the nearest example
            Ypred[i] = self.ytr[mode(predicted_list)] 

        return Ypred

# Let us create a validation set with 1000 examples
Xval_row = Xtr_row[:1000, :]
Yval_row = Ytr[:1000]

Xtr_row = Xtr_row[1000:, :]
Ytr = Ytr[1000:]

# find the hyper-parameter that works the best

validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:

    # use the particular value of k and evaluate on validation data
    nn = NearestNeighbor()
    nn.train(Xtr_row, Ytr)

    Yval_predict = nn.predict(Xval_row, k=k)
    acc = np.mean(Yval_predict == Yval)
    print("accuracy: %f" %(acc,))

    # keep track of what works on the validation set
    validation_accuracies.append((k, acc))

print(validation_accuracies)
# use k for which the maximum accuracy is observed

"""
Improved Technique: 
    
    CROSS VALIDATION:

    The idea is that instead of arbitrarily picking the first 1000 datapoints 
    to be the validation set and rest training set, you can get a better and less
    noisy estimate of how well a certain value of k works by iterating over
    different validation sets and averaging the performance across these.
    For example, in 5-fold cross-validation, we would split the training data into 
    5 equal folds, use 4 of them for training, and 1 for validation. We would then
    iterate over which fold is the validation fold, evaluate the performance, and 
    finally average the performance across the different folds.Typical number of 
    folds you can see in practice would be 3-fold, 5-fold or 10-fold cross-validation.
 
 """
