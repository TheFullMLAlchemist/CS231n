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
        W = Wtry
        bestloss = loss
    print("iter %d loss is %f" % (i, bestloss))


# observation 21.4% accuracy score

# Strategy-3: Following the Gradient

"""
We use first principle to evaluate the gradient
We first use numerical method or first principle to evaluate the
gradient

f'(x) = lim   (f(x+h) - f(x)) / h
        h -> 0

we use a very small value of h for our calculation
"""

def eval_numerical_gradient(f, x):
    """
    A naive implementation of numerical gradient of f at x
    - f should be a function that takes a single arguement
    - x is the point (numpy array) to evaluate the graadient at
    """

    fx = f(x) # Evaluate function value at original point
    grad = np.zeros(x.shape)
    h = .00001

    # iterate over all indexes in x
    it = np.nditer(x, flags = ['multi_index'], op_flag = ["readwrite"])
    while not it.finished:

        # evaluate function at x + h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value+h # increment by x
        fxh = f(x) # evaluate at x+h

        x[ix] = old_value # restore to previous value (important step)

        # Compute the partial derivative
        grad[ix] = (fxh - fx) / h # the slope
        it.iternext() # step to next dimension

    return grad

"""
We sometimes encounter non-differentiable points while calculating 
gradient, so to over come that problem we use sub-gradient

otherwise numerically we prefer central difference method

    f'(x) = lim (f(x+h) - f(x-h)) / 2h
            h -> 0
"""


def CIFAR10_loss_fun(W):
    return L(X_train, Y_train, W)

W = np.random.rand(10, 3073) * .001 # Random weight

df = eval_numerical_gradient(CIFAR10_loss_fun, W) # get the gradient

loss_original = CIFAR10_loss_fun(W) # the original loss

print("original loss %f" %(loss_original))

for step_size_log in range(-10, 0):
    step_size = 10**step_size_log
    W_new = W - step_size*df # new position in the weight space
    loss_new  = CIFAR10_loss_fun(W_new)
     print("for step size %f new loss: %f" % (step_size, loss_new))

"""
In the code above that computes W_new, we are making an update in the negative
direction of the gradient df since we wish our loss function to decrease, not increase
Covex optimization
"""

# Gradient Descent

while True:
    weights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += -step_size*weights_grad # performance parameter update

# Vanilla Mini-Batch Gradient Descent 

while True:
    data_batch = sample_training_data(data, 256) # sample 256 examples
    weights_grad = eval_gradient(loss_fun, data_batch, weights)
    weights += -step_size*weights_grad # perform parameter update


"""
Due to memory limitations we sometimes cannot compute gradient or weights for large datasets
therefore to overcome this problem we use mini-batches for certain sizes.

Generally the size of mini-batches are in the power of 2 as it was observed that 
computation is much more optimal when we do this.

When the mini-batch size is equal to 1 then it is know as Stochastic Gradient Descent
sometimes often refered to as on-line Gradient Descent. It is not much computationally 
optimal as the optimality obtainted from vectorized implementation is lost

"""








