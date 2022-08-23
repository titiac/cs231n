from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    # FullyConnectedNet是 object的子类，是关于新式类和经典类 *了解即可
    """Class for a multi-layer fully connected neural network.
    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options.
    For a network with L layers, the architecture will be
    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.
    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    #总共有L+1层，含第0层(输入层，不用接权重)，最后一层编号为L层(输出层)，共有L-1层隐藏层

    def __init__(
            self,
            hidden_dims,
            input_dim=3 * 32 * 32,
            num_classes=10,
            dropout_keep_ratio=1,
            normalization=None,
            reg=0.0,
            weight_scale=1e-2,
            dtype=np.float32,
            seed=None,
    ):
        """Initialize a new FullyConnectedNet.
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
            # hidden_dims隐匿层大小的列表
        - input_dim: An integer giving the size of the input.
            # input_dim给出一个输入大小的整数，相当于一个数据对象（单张图片）的数据规模
        - num_classes: An integer giving the number of classes to classify.
            # num_classes给出类的数量的一个整数
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            # dropout_keep_ratio 一个范围为0~1的数，给出了dropout的强度，随机失活
            If dropout_keep_ratio=1 then the network should not use dropout at all.
            # 如果dropout_keep_ratio值为1，则不用使用dropout
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
            # 使用什么类型的归一化, 有三个取值 “batchnorm”批量归一化“layernorm”层标准化“None”不做归一化
            layernorm —— 将各层转化为标准正态分布，加快训练速度， 加速收敛
        - reg: Scalar giving L2 regularization strength.
            # 给L2正则化的系数
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
            # 标量， 给出权重随机初始化的标准差
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
            # numpy的数据类型对象，有两个float32和float64， 后者精度更高
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
            #如果这个seed不是空的话，就将其传递给dropout层，这将使dropout层确定（可预见性），更好的进行梯度检查
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1 # 如果不等就返回True，就使用dropout
        self.reg = reg # 正则化强度
        self.num_layers = 1 + len(hidden_dims) # 得出层数
        self.dtype = dtype # 精度，一般为 np.float64
        self.params = {} # 可学习参数放这

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        """初始化网络的参数，将所有值存储在self中。params字典。在W1和b1中存储第一层的权重和偏差;第二层使用W2、b2等。
          权重应该从一个以0为中心的正态分布初始化，且标准差等于weight_scale。
          偏差应该初始化为零。使用批归一化时，在gamma1和beta1中存储第一层的scale和shift参数;
          第二层使用gamma2和beta2等。Scale参数应该初始化为1，而shift参数应该初始化为0。"""
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # 通过循环初始化权重
        for i in range(self.num_layers):
            if i == 0: # 第一个隐匿层（接收输出层的层）
                w = np.random.normal(0.0, weight_scale, size=(input_dim, hidden_dims[i]))
                b = np.zeros(hidden_dims[i])
                self.params['W' + str(i + 1)] = w # 这种添加字典值可以学习一下
                self.params['b' + str(i + 1)] = b
            elif i == (self.num_layers - 1): # 最后一层也就是输出层
                w = np.random.normal(0.0, weight_scale, size=(hidden_dims[i - 1], num_classes))
                b = np.zeros(num_classes)
                self.params['W' + str(i + 1)] = w
                self.params['b' + str(i + 1)] = b
            else: # 只有上面两种情况是特殊的，其余都可以用下面的方式概括
                w = np.random.normal(0.0, weight_scale,size=(hidden_dims[i - 1], hidden_dims[i]))
                b = np.zeros(hidden_dims[i])
                self.params['W' + str(i + 1)] = w
                self.params['b' + str(i + 1)] = b
            # print(w.shape)#可删

        #列表里面不能存储array对象，字典里面可以存
        # 生成层
        # self.layers = OrderedDit()
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        #  训练时，可认为随机失活是对完整神经网络抽样出一些子集，只更新随机失活的子网络参数
        #  在测试则不用随机失活
        self.dropout_param = {} # 随机失活的参数
        if self.use_dropout: # 如果使用随机失活
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None: # 随机数种子，可预见性，如果是None那就是完全随机的
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)] #列表中包含num_layers-1个{"mode": "train"} ——复合类型
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)] # 参考博客 组织列表——列表解析

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype) # 将参数数据精度从默认的float32 -> float64

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype) # float64
        mode = "test" if y is None else "train"
        # 如果y非空则mode为train，否则为test
        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout: # 如果使用随机失活
            self.dropout_param["mode"] = mode #当前的训练类型 train/test
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        caches = {} # 需要给予bp

        for i in range(self.num_layers): # 如果没有引入激活函数，只是线性的，那么根据矩阵乘法的性质，多层可以等价于一层的神经网络
            if i == (self.num_layers - 1):
                #如果是最后一层，就直接affine
                scores, caches[self.num_layers - 1] = affine_forward(X, self.params['W' + str(self.num_layers)],
                                                                     self.params['b' + str(self.num_layers)])
            else:
                #如果不是最后一层，用affine加relu
                X, caches[i] = affine_relu_forward(X, self.params['W' + str(i + 1)], self.params['b' + str(i + 1)])
                # 此处的X是中间变量，不关心 前n-1层caches = (affcin_caches, relu_caches)


        # print(len(caches[0]))#为啥是2?
        # AR1_out, AR1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        # A2_out, A2_cache = affine_forward(AR1_out, self.params['W2'], self.params['b2'])
        # scores = X
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        """待办事项: 
        实现全连接神经网络的反向传播。将损失存储在梯度词典中的损失变量和梯度中。使用 softmax 计算数据丢失，即采用交叉熵损失，
        并确保梯度[ k ]持有自己的梯度。不要忘记添加 L2正规化！在使用批处理/层规范化时，不需要规范化缩放和移位参数。
        注意: 为了确保您的实现与我们的实现相匹配，并且您通过了自动化测试，请确保您的 L2正则化包含一个0.5的因子，以简化梯度的表达式。
        """
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss, dscores = softmax_loss(scores, y)
        for i in range(self.num_layers):
            loss += 0.5 * self.reg * np.sum(self.params['W' + str(i + 1)] * self.params['W' + str(i + 1)])

        for i in range(self.num_layers):
            if i == 0:
                #如果是最后一层，算affine的backward
                dx, dw, db = affine_backward(dscores, caches[self.num_layers - 1 - i])
                dw += 0.5 * 2 * self.reg * self.params['W' + str(self.num_layers - i)]
                grads['W' + str(self.num_layers - i)] = dw
                grads['b' + str(self.num_layers - i)] = db
            else:
                #如果不是最后一层，算affine加relu的backward
                dx, dw, db = affine_relu_backward(dx, caches[self.num_layers - 1 - i])
                dw += 0.5 * 2 * self.reg * self.params['W' + str(self.num_layers - i)]
                grads['W' + str(self.num_layers - i)] = dw
                grads['b' + str(self.num_layers - i)] = db

        # dX2, dW2, db2 = affine_backward(dsm, A2_cache)
        # dW2 += 0.5 * 2 * self.reg * self.params['W2']#正则化后面还有关于W1、W2的加项
        # grads['W2'] = dW2
        # grads['b2'] = db2
        # dX1, dW1, db1 = affine_relu_backward(dX2, AR1_cache)
        # dW1 += 0.5 * 2 * self.reg * self.params['W1']#正则化后面还有关于W1、W2的加项
        # grads['W1'] = dW1
        # grads['b1'] = db1
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads