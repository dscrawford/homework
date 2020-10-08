################################################################################
#
# LOGISTICS
#
#    Daniel Crawford
#    dsc160130
#
# FILE
#
#    nn.py
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
#    1. A summary of my nn.py code:
#
#       Forward Path Code:
#           The forward path is controlled by template classes called Layer, WeightedLayer and Model.
#           Additionally, the forward pass will save input if needed for the backward pass.
#               Model simply iterates through each of the layers and pumps output from previous layer to next layer.
#               Layer is a non-weighted NN layer that will pass forward an arbitrary non-weighted operation (Max pool, vectorize, ReLU)
#               WeightedLayer is a weighted NN layer that pass forward an arbitrary weigthed operation (MatrixMultiplication, Addition)
#
#       Error Code:
#           Error is computed through cross_entropy_loss, the displayed loss will usually be this function aggregated
#           and averaged.
#
#       Backward Path Code:
#           Similar to forward path, the backward path goes through the model through the Model, Layer and WeightedLayer classes.
#               Model backward function takes correct label for data and puts it into last layer.
#               Layer+WeightedLayer return their respective derivatives multiplied by the derivatives they receive.
#
#       Weight Update Code:
#           Each WeightedLayer holds an update weight called update_weights same dimensions as weights. Upon calling
#           layer.update(), update_weights are added to the weights with an input learning rate.
#
#       Extra:
#           You will notice in the backwards pass that Normalize and Vectorize do not return anything. This is for
#           performance only because they are not needed for this specific example.
#
#    2. Accuracy display
#
# Train Loss: 0.292635: 100%|██████████| 60000/60000 [04:59<00:00, 200.51it/s]
# Train Loss: 0.292635275128901
# Test Accuracy: 0.9546
# Train Loss: 0.090328: 100%|██████████| 60000/60000 [04:55<00:00, 203.24it/s]
# Train Loss: 0.09032760041587834
# Test Accuracy: 0.9715
# Train Loss: 0.057369: 100%|██████████| 60000/60000 [04:56<00:00, 202.23it/s]
# Train Loss: 0.0573690924060683
# Test Accuracy: 0.9695
# Train Loss: 0.04139: 100%|██████████| 60000/60000 [04:55<00:00, 203.16it/s]
# Train Loss: 0.04139010430699511
# Test Accuracy: 0.9781
# Train Loss: 0.031412: 100%|██████████| 60000/60000 [05:03<00:00, 197.48it/s]
# Train Loss: 0.03141150109060862
# Test Accuracy: 0.9707
# Train Loss: 0.026368: 100%|██████████| 60000/60000 [04:53<00:00, 204.29it/s]
# Train Loss: 0.02636792131513013
# Test Accuracy: 0.9738
# Train Loss: 0.021671: 100%|██████████| 60000/60000 [04:52<00:00, 205.18it/s]
# Train Loss: 0.0216710897789739
# Test Accuracy: 0.9776
# Train Loss: 0.019675: 100%|██████████| 60000/60000 [04:48<00:00, 207.96it/s]
# Train Loss: 0.019674705079921723
# Test Accuracy: 0.9774
# Train Loss: 0.017662: 100%|██████████| 60000/60000 [04:46<00:00, 209.41it/s]
# Train Loss: 0.017662136397378944
# Test Accuracy: 0.9692
# Train Loss: 0.01757: 100%|██████████| 60000/60000 [04:45<00:00, 209.93it/s]
# Train Loss: 0.01757029891800587
# Test Accuracy: 0.9812
#
#    3. Performance display
#
# Normalize Layer
# Input dim:(28, 28)
# Output dim:(28, 28)
# Num Parameters: 0
# MACs: 784
# Vectorizer Layer
# Input dim:(28, 28)
# Output dim:(1, 784)
# Num Parameters: 0
# MACs: 784
# Matrix Multiplication Layer
# Input dim:[1, 784]
# Output dim:[1, 1000]
# Num Parameters: 784000
# MACs: 784000
# Addition Layer
# Input dim:[1, 1000]
# Output dim:[1, 1000]
# Num Parameters: 1000
# MACs: 1000
# ReLU Layer
# Input dim:(1, 1000)
# Output dim:(1, 1000)
# Num Parameters: 0
# MACs: 1000
# Matrix Multiplication Layer
# Input dim:[1, 1000]
# Output dim:[1, 100]
# Num Parameters: 100000
# MACs: 100000
# Addition Layer
# Input dim:[1, 100]
# Output dim:[1, 100]
# Num Parameters: 100
# MACs: 100
# ReLU Layer
# Input dim:(1, 100)
# Output dim:(1, 100)
# Num Parameters: 0
# MACs: 100
# Matrix Multiplication Layer
# Input dim:[1, 100]
# Output dim:[1, 10]
# Num Parameters: 1000
# MACs: 1000
# Addition Layer
# Input dim:[1, 10]
# Output dim:[1, 10]
# Num Parameters: 10
# MACs: 10
# ReLU Layer
# Input dim:(1, 10)
# Output dim:(1, 10)
# Num Parameters: 0
# MACs: 10
# Test Accuracy:  98.12
# Test Loss:  0.08622763789432848
# Time Taken:  2946.323134660721
#
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

# MY IMPORTS
from time import time
from tqdm import tqdm

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
    return -1 * np.log(x[x_true])


def display_layer_info(type, input_size, output_size, parameter_size, MACs):
    return type + ' Layer\n' + 'Input dim:' + str(input_size) + '\nOutput dim:' + str(output_size) + \
           '\nNum Parameters: ' + str(parameter_size) + '\nMACs: ' + str(MACs) + '\n'


# Layer
# Class which consists of a forward, backward and update function (by default, update does nothing)
class Layer:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        pass

    def forward(self, input):
        self.output = input
        return self.output

    def backward(self, derivative):
        return derivative

    def update(self, lr):
        return

    def description(self):
        return display_layer_info('Standard', self.input_dim, self.input_dim, 0, 0)


# WeightedLayer
# Class which has weighted operations associated with it
class WeightedLayer(Layer):
    def __init__(self, input_dim, weight_dim, init_method='uniform'):

        if init_method not in {'uniform', 'normal', 'zero'}:
            print('ERROR: Invalid init_method selection')
            exit(1)

        self.weight_dim = weight_dim

        super().__init__(input_dim)
        if init_method == 'uniform':
            self.weights = np.random.uniform(0, 1, weight_dim) / 100
        elif init_method == 'normal':
            self.weights = np.random.normal(0, 1, weight_dim) / 100
        else:
            self.weights = np.zeros(weight_dim)
        self.update_weights = np.zeros(weight_dim)

    def forward(self, input):
        return super().forward(input)

    def backward(self, derivative):
        return super().backward(derivative)

    def update(self, lr):
        self.weights = self.weights - lr * self.update_weights
        self.update_weights = np.zeros(self.weight_dim)

    def description(self):
        return display_layer_info('Weighted', self.input_dim, self.input_dim, len(self.weights), 0)


# Normalize
# Layer which normalizes all values from a given input with a constant
class Normalize(Layer):
    def __init__(self, input_dim, norm_constant):
        super().__init__(input_dim)
        self.norm_constant = norm_constant

    def forward(self, input):
        return super().forward(input / self.norm_constant)

    def backward(self, derivative):
        # return derivative * 255.0
        return None

    def description(self):
        return display_layer_info('Normalize', self.input_dim, self.input_dim, 0, np.product(self.input_dim))


# Vectorizer
# Layer which flattens input dimensions (i.e, a (32,16,16) becomes a (1, 32 * 16 * 16)
class Vectorizer(Layer):
    def __init__(self, input_dim):
        super().__init__(input_dim)
        self.output_dim = (1, np.product(input_dim))

    def forward(self, input):
        return super().forward(np.reshape(input, self.output_dim))

    def backward(self, derivative):
        # return np.reshape(derivative, self.input_dim)
        return None

    def description(self):
        return display_layer_info('Vectorizer', self.input_dim, self.output_dim, 0, np.product(self.input_dim))


# MatrixMultiplication
# Weighted Layer which transforms a (n,m) matrix to a (n, a) matrix with a (m,a) matrix. (i.e: (n,m) * (m,a) -> (n,a))
# Matrices can be n-dimensional
class MatrixMultiplication(WeightedLayer):
    def __init__(self, input_dim, output_dim, init_method='uniform'):
        self.input_dim = cast_dim(input_dim)
        self.output_dim = cast_dim(output_dim)
        self.h_dim = (input_dim[-1], output_dim[-1])
        super().__init__(self.input_dim, self.h_dim, init_method)

    def forward(self, input):
        self.input = input
        return super().forward(np.matmul(input, self.weights))

    def backward(self, derivative):
        self.update_weights += np.matmul(np.transpose(self.input), derivative)
        return np.matmul(derivative, np.transpose(self.weights))

    def description(self):
        return display_layer_info('Matrix Multiplication', self.input_dim, self.output_dim, np.product(self.weights.shape),
                                  np.product(self.input_dim) * np.product(self.output_dim[1:]))


# Addition
# Weigthed layer that simply adds a (n,m) weighted bias to an (n,m) matrix
# Matrices can be n-dimensional
class Addition(WeightedLayer):
    def __init__(self, input_dim, init_method='uniform'):
        input_dim = cast_dim(input_dim)
        self.h_dim = input_dim

        super().__init__(input_dim, self.h_dim, init_method)

    def forward(self, input):
        return super().forward(self.weights + input)

    def backward(self, derivative):
        self.update_weights += derivative
        return derivative

    def description(self):
        return display_layer_info('Addition', self.input_dim, self.input_dim, np.product(self.weights.shape), np.product(self.input_dim))


# ReLU
# Layer which applies simple ReLU function elementwise to the matrix
class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        self.input_dim = input.shape
        self.input = input
        return super().forward(np.maximum(input, 0))

    def backward(self, derivative):
        return derivative * (self.input > 0)

    def description(self):
        return display_layer_info('ReLU', self.input_dim, self.input_dim, 0, np.product(self.input_dim))


# SoftMax
# Layer which applies softmax function to input matrix.
class SoftMax(Layer):
    def __init__(self):
        pass

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def forward(self, input):
        self.input_dim = input.shape
        return super().forward(self.softmax(input))

    def backward(self, x_true):
        derivative = self.output
        derivative[0][x_true] -= 1
        return derivative

    def description(self):
        return display_layer_info('ReLU', self.input_dim, self.input_dim, 0, np.product(self.input_dim))


# Model
# Consists of layers and a for loop which iterates through each layer and pumps output forward and backward.
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

    def backward(self, back):
        for layer in reversed(self.layers):
            back = layer.backward(back)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    def description(self):
        for layer in self.layers:
            print(layer.description())


model = Model(init_method='normal')

t = time()
train_loss = []

for epoch in range(NUM_EPOCHS):
    lr = LEARNING_RATE
    loss = 0

    tq = tqdm(range(len(train_data)))
    for i in tq:
        d = train_data[i]
        label = train_labels[i]
        predict = model.forward(d)
        loss += cross_entropy_loss(predict, label)
        model.backward(label)
        model.update(lr)
        tq.set_description('Train Loss: ' + str(np.round(loss / (i + 1), 6)))

    train_loss.append(loss)

    num_correct = 0
    for d, label in zip(test_data, test_labels):
        predict = model.forward(d)
        num_correct += (np.argmax(predict) == label)

    accuracy = num_correct / len(test_data)

    print('Train Loss: ' + str(loss / len(train_data)))
    print('Test Accuracy: ' + str(accuracy))
    print()

time_taken = time() - t
#
test_predicted_labels = []
test_loss = 0
num_correct = 0
for d, label in zip(test_data, test_labels):
    predict = model.forward(d)
    test_loss += cross_entropy_loss(predict, label)
    test_predicted_labels.append(np.argmax(predict))
    num_correct += (np.argmax(predict) == label)

accuracy = num_correct / len(test_data)

# cycle through the training data
# forward pass
# loss
# back prop
# weight update

# cycle through the testing data
# forward pass
# accuracy

# per epoch display (epoch, time, training loss, testing accuracy, ...)
model.description()

################################################################################
#
# DISPLAY
#
################################################################################

# accuracy display
print('Test Accuracy: ', 100 * num_correct / len(test_data))
# final value
print('Test Loss: ', test_loss / len(test_data))
# plot of accuracy vs epoch
plt.plot(list(range(0, NUM_EPOCHS)), train_loss)
plt.xlabel('EPOCH')
plt.ylabel('Cross Entropy Loss')
plt.title('Neural Network Performance Chart')
plt.show()

# performance display
# total time
print('Time Taken: ', time_taken)
# per layer info (type, input size, output size, parameter size, MACs, ...)

# example display
# replace the xNN predicted label with the label predicted by the network

fig = plt.figure(figsize=(DISPLAY_COL_IN, DISPLAY_ROW_IN))
ax = []
for i in range(DISPLAY_NUM):
    img = test_data[i, :, :, :].reshape((DATA_ROWS, DATA_COLS))
    ax.append(fig.add_subplot(DISPLAY_ROWS, DISPLAY_COLS, i + 1))
    ax[-1].set_title('True: ' + str(test_labels[i]) + ' xNN: ' + str(test_predicted_labels[i]))
    plt.imshow(img, cmap='Greys')
plt.show()
