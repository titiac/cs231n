from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(X.shape[0]):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        p = np.exp(scores) / np.sum(np.exp(scores))
        for j in range(W.shape[1]):
            if j == y[i]:
                dW[:, j] += (p[j] - 1) * X[i]
            else:
                dW[:, j] += p[j] * X[i]
        loss += -np.log(p[y[i]])
    loss /= X.shape[0]
    loss += reg * np.sum(W ** 2)

    dW /= X.shape[0]
    dW += 2.0 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True) # 对数值进行平移，使最大值为0，避免计算softmax函数造成数值爆炸
    p = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True) # 这个已经转化为概率矩阵了
    loss = np.sum(-np.log(p[np.arange(X.shape[0]), y])) / X.shape[0]
    p[np.arange(X.shape[0]), y] = p[np.arange(X.shape[0]), y] - 1 # 不懂的地方
    dW = np.dot(X.T, p) # 不懂的地方 
    loss += reg * np.sum(W ** 2) #正则化

    dW = dW / X.shape[0] + 2.0 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW