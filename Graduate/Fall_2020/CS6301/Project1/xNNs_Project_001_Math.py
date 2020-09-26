################################################################################
#
# LOGISTICS
#
#    Daniel Crawford
#    dsc160130
#
# DESCRIPTION
#
#    MNIST image classification with an xNN written and trained in Python
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Run all
#
# NOTES
#
#    1. This does not use PyTorch, TensorFlow or any other xNN library
#
#    2. Include a short summary here in nn.py of what you did for the neural
#       network portion of code
#
#    3. Include a short summary here in cnn.py of what you did for the
#       convolutional neural network portion of code
#
#    4. Include a short summary here in extra.py of what you did for the extra
#       portion of code
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

#
# you should not need any import beyond the below
# PyTorch, TensorFlow, ... is not allowed
#

import os.path
import urllib.request
import gzip
import math
import numpy             as np
import matplotlib.pyplot as plt

################################################################################
#
# PARAMETERS
#
################################################################################

# hyper parameters
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
STABILIZING_CONSTANT = 1e-5

# data
DATA_NUM_TRAIN = 60000
DATA_NUM_TEST = 10000
DATA_CHANNELS = 1
DATA_ROWS = 28
DATA_COLS = 28
DATA_CLASSES = 10
DATA_URL_TRAIN_DATA = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA = 'test_data.gz'
DATA_FILE_TEST_LABELS = 'test_labels.gz'

# display
DISPLAY_ROWS = 8
DISPLAY_COLS = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM = DISPLAY_ROWS * DISPLAY_COLS

################################################################################
#
# DATA
#
################################################################################

# download
if not os.path.exists(DATA_FILE_TRAIN_DATA):
    urllib.request.urlretrieve(DATA_URL_TRAIN_DATA, DATA_FILE_TRAIN_DATA)
if not os.path.exists(DATA_FILE_TRAIN_LABELS):
    urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if not os.path.exists(DATA_FILE_TEST_DATA):
    urllib.request.urlretrieve(DATA_URL_TEST_DATA, DATA_FILE_TEST_DATA)
if not os.path.exists(DATA_FILE_TEST_LABELS):
    urllib.request.urlretrieve(DATA_URL_TEST_LABELS, DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN * DATA_ROWS * DATA_COLS)
train_data = np.frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data = train_data.reshape(DATA_NUM_TRAIN, 1, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels = np.frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST * DATA_ROWS * DATA_COLS)
test_data = np.frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data = test_data.reshape(DATA_NUM_TEST, 1, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels = np.frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)


# debug
# print(train_data.shape)   # (60000, 1, 28, 28)
# print(train_labels.shape) # (60000,)
# print(test_data.shape)    # (10000, 1, 28, 28)
# print(test_labels.shape)  # (10000,)

################################################################################
#
# YOUR CODE GOES HERE
#
################################################################################

#
# feel free to split this into some number of classes, functions, ... if it
# helps with code organization; for example, you may want to create a class for
# each of your layers that store parameters, performs initialization and
# includes forward and backward functions
#

def cast_dim(dim):
    return (dim,) if type(dim) != tuple and type(dim) != list else list(dim)

def cross_entropy_loss(x, x_true):
    return -1 * np.log(x[x_true] + STABILIZING_CONSTANT)


class Layer:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        pass

    def forward(self, input):
        pass

    def backward(self, loss):
        pass


class WeightedLayer(Layer):
    def __init__(self, input_dim, weight_dim, init_method='uniform'):

        if init_method not in {'uniform', 'normal', 'zero'}:
            print('ERROR: Invalid init_method selection')
            exit(1)

        super().__init__(input_dim)
        if init_method == 'uniform':
            self.weights = np.random.uniform(0, 1, weight_dim)
        elif init_method == 'normal':
            self.weights = np.random.normal(0, 1, weight_dim)
        elif init_method == 'zero':
            self.weights = np.zeros(weight_dim)

    def forward(self, input):
        return super().forward(input)

    def backward(self, loss):
        return super().backward(loss)


class Normalize(Layer):
    def __init__(self, input_dim, norm_constant):
        super().__init__(input_dim)
        self.norm_constant = norm_constant

    def forward(self, input):
        return input / self.norm_constant

    def backward(self, loss):
        return np.reshape(loss, self.input_dim)

class Vectorizer(Layer):
    def __init__(self, input_dim):
        super().__init__(input_dim)
        self.output_dim = (1, np.product(input_dim))

    def forward(self, input):
        return np.reshape(input, self.output_dim)

    def backward(self, loss):
        return np.reshape(loss, self.input_dim)


class MatrixMultiplication(WeightedLayer):
    def __init__(self, input_dim, output_dim, init_method='uniform'):
        input_dim = cast_dim(input_dim)
        output_dim = cast_dim(output_dim)
        self.h_dim = (input_dim[-1], output_dim[-1])
        super().__init__(self, self.h_dim, init_method)

    def forward(self, input):
        return np.matmul(input, self.weights)

    def backward(self, loss):
        return 0


class Addition(WeightedLayer):
    def __init__(self, input_dim, init_method='uniform'):
        input_dim = cast_dim(input_dim)
        self.h_dim = input_dim

        super().__init__(self, self.h_dim, init_method)

    def forward(self, input):
        return self.weights + input

    def backward(self, loss):
        return 0


class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        return np.maximum(input, 0)

    def backward(self, loss):
        return 0


class SoftMax(Layer):
    def __init__(self):
        pass

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def forward(self, input):
        return np.array([self.softmax(x) for x in input])

    def backward(self, true):
        return


class Model:
    def __init__(self, init_method='uniform'):
        self.layers = [
            Normalize((28, 28), 255.0),
            Vectorizer((28, 28)),
            MatrixMultiplication((1, 28 ** 2), (1, 1000), init_method=init_method),
            Addition((1, 1000), init_method=init_method),
            ReLU(),
            MatrixMultiplication((1, 1000), (1, 100), init_method=init_method),
            Addition((1, 100), init_method=init_method),
            ReLU(),
            MatrixMultiplication((1, 100), (1, 10), init_method=init_method),
            Addition((1, 10), init_method=init_method),
            SoftMax()
        ]
        self.loss = cross_entropy_loss

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output[0]

    def backward(self, true):
        for layer in reversed(self.layers):
            loss = layer.backward(loss)



model = Model()
print(cross_entropy_loss(model.forward(train_data[0]), train_labels[0]))
# for epoch in range(1):
#     lr = LEARNING_RATE
#
#     from time import time
#     t = time()
#     for d in train_data:
#         model.forward(d)
#     print(time() - t)
# cycle through the training data
# forward pass
# loss
# back prop
# weight update

# cycle through the testing data
# forward pass
# accuracy

# per epoch display (epoch, time, training loss, testing accuracy, ...)

################################################################################
#
# DISPLAY
#
################################################################################

#
# more code for you to write
#

# accuracy display
# final value
# plot of accuracy vs epoch

# performance display
# total time
# per layer info (type, input size, output size, parameter size, MACs, ...)

# example display
# replace the xNN predicted label with the label predicted by the network

# fig = plt.figure(figsize=(DISPLAY_COL_IN, DISPLAY_ROW_IN))
# ax = []
# for i in range(DISPLAY_NUM):
#     img = test_data[i, :, :, :].reshape((DATA_ROWS, DATA_COLS))
#     ax.append(fig.add_subplot(DISPLAY_ROWS, DISPLAY_COLS, i + 1))
#     ax[-1].set_title('True: ' + str(test_labels[i]) + ' xNN: ' + str(test_labels[i]))
#     plt.imshow(img, cmap='Greys')
# plt.show()
