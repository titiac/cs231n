from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from ..classifiers.linear_svm import *
from ..classifiers.softmax import *
from past.builtins import xrange


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        Train this linear classifier using stochastic gradient descent. 随机梯度下降
        训练方法：随机梯度下降 SDG
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization. 学习速率也就是学习步长，每次改变梯度的程度
        - reg: (float) regularization strength.  正则强度
        - num_iters: (integer) number of steps to take when optimizing   进行优化所需要的步骤数
        - batch_size: (integer) number of training examples to use at each step. 在每个步骤中使用的训练例子的数量
        - verbose: (boolean) If true, print progress during optimization. 如果是true就在优化过程这种打印进度

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape    # X_trian 49000 * 3073
        num_classes = (
            np.max(y) + 1
        )  # assume y takes values 0...K-1 where K is number of classes 返回一个元组 return A tuple
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)   # 如果没有提供权重举证，则通过随机初始化产生一个

        # Run stochastic gradient descent to optimize W
        loss_history = []  # 记录每次迭代得到的损失值
        for it in range(num_iters):   # 迭代
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            idx = np.random.choice(num_train, size=batch_size, replace=True)
            X_batch = X[idx, :]
            y_batch = y[idx]
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.W -= learning_rate * grad # 随机梯度下降
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        scores = np.dot(X, self.W)  # 使用得到的权重，计算得分
        y_pred = np.argmax(scores, axis=1) # 从每个对象在每个类的得分中找到最大值就是其在当下最可能的标签

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):  # linearClassifier的子类
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg): # 重构上面的loss方法
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg) # 矢量化的方法求损失和梯度, 会比未矢量化的快


class Softmax(LinearClassifier):    # linearClassifier的子类
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)