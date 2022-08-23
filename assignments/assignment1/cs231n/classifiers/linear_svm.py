from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    建立一个svm 损失函数， 本地执行（在循环基础上）
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    输入我们对小批量进行操作的N个例子（D 维度， C类）。
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.   包含权重的
    - X: A numpy array of shape (N, D) containing a minibatch of data.   包含小部分数据的
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]  # 10
    num_train = X.shape[0]  # 500
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)  # 评分函数 (1 *3073) * (3073 * 10) = 1 * 10  点积
        correct_class_score = scores[y[i]]  # 正确的类的得分, y[i] 第i个元素的标签标号
        for j in range(num_classes):  # 10个种类， 0 到 9遍历 scores
            if j == y[i]:  # 如果y[i] = j 那么就不处理，因为这个就是正确分类的得分
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1， 一般delta都设置为1， 这个效果不错
            if margin > 0:  # max（0, margin）
                loss += margin  # 当得分margin 比 0大就加入损失值
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train  # 要计算平均损失值，所以有除以总项数

    # Add regularization to the loss. 正则化损失
    loss += reg * np.sum(W * W)  # loss += reg * np.sum(np.square(W)) 这是最常用的正则化惩罚——L2 范式
    # reg 一般无法直接取得，需要进行交叉验证取得
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW /= (1.0 * num_train)
    dW += 2.0 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X.dot(W)
    correct_class_score = scores[np.arange(num_train), y]  # different with scores[:, y]
    # 将每行的正确的类的分数添加进 correct_class_score
    margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + 1)
    # newaxis 为numpy数组添加一个维度， 这里相当于将一维数组转变为二维数组, 就变成 500 * 1的二维数组
    margins[np.arange(num_train), y] = 0
    # 将正确分类的分数置为0
    loss += np.sum(margins)  # 对margins 数组内的所有元素求和， 也就是完成数据损失的计算
    loss /= num_train  # 对数据损失和求平均
    loss += reg * np.sum(W * W)  # 添加正则化损失

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    margins[margins > 0] = 1.0   
    num_to_loss = np.sum(margins, axis=1)

    margins[np.arange(num_train), y] = -num_to_loss
    dW = np.dot(X.T, margins)
    dW /= num_train
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
