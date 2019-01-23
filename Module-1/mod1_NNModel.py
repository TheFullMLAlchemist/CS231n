import numpy as np 
import matplotlib.pyplot as plt 


# Generating some random data
N = 200 # Number of points per class
D = 2 # Dimentionality
K = 3 # Number of classes
X = np.zeros((N*K, D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels

for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N) # radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random().randn(N)*0.2 #theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

# Visualization of data
plt.scatter(plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral))
plt.show()

# Training a Softmax Classfier 
# initialize parameter randomly
W = 0.01*np.random.randn(D, K)
b = np.zeros((1, K))

# compute class scores for a linear classifier
scores = np.dot(X, W) + b

num_examples = X.shape[0]
exp_scores = np.exp(scores)
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

correct_logprobs = -np.log(probs[range(num_examples), y])

# Compute the loss: average cross entropy loss and regularization
data_loss = np.sum(correct_logprobs)/num_examples
reg_loss = 0.5*reg*np.sum(W*W)
loss = data_loss + reg_loss

dscore = probs
dscore[range(num_examples), y] -= 1
dscore /= num_examples

dW = np.dot(X.T, dscores)
db = np.sum(dscores, axis=0, keepdims=True)
dw += reg*W


# Performing parameter update
W += -step_size * dW
b += -step_size * db

# Training a softmax classofier

W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))

step_size = 1e-0
reg = 1e-3

# gradient descent loop
num_examples = X.shape[0]

for i in range(200):
    scores = np.dot(X, W) + b
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
    # compute the loss: average cross-entropy loss and regularization
    correct_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(correct_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss
     if i % 10 == 0:
        print "iteration %d: loss %f" % (i, loss)
  
    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples
  
    # backpropate the gradient to the parameters (W,b)
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
  
    dW += reg*W # regularization gradient
  
     # perform a parameter update
    W += -step_size * dW
    b += -step_size * db

# evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))


