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

# Define a Nearest Neighbor classifier class
class NearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self, X, y):
        # X is N x D where each row is an example. Y is 1D of size N
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        # X is N x D where each row is an example we wish to predict label for
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all test rows
        for i in xrange(num_test):
            # find the nearest training image to the ith test image
            # using the L1 distance (sum of absolute value difference)
            distances = np.sum(np.abs(self.Xtr - X[i:]), axis=1)
            # Get the index with smallest distance
            min_index = np.argmin(distances)
            # predict the label of the nearest example
            Ypred[i] = self.ytr[min_index] 

        return Ypred

# Create a Nearest Neighbor classifier class
nn = NearestNeighbor()
# Train the classifier on the training image and labels 
nn.train(Xtr_row, Ytr)
# predit label on the test images
Yte_predict = nn.predict(Xte_row)
# Print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)

print("accuracy: %f" %(np.mean(Yte_predict == Yte)))


# The accuracy achieved by this code on CIFAR10 is 38.6%

""" Changes that can be done:
The choice of distance: L2 distance which is 
square root of sum of square of distance between the pixels 

distances = np.sqrt(np.sum(np.square(self.Xtr - X[i:]), axis = 1))

This gives the accuracy of 35.4%

Other distance choices are:
1-norm which is manhattan distance i.e. horizontal distance plus vertical distance
2-norm which is simply Euclidian Distance

