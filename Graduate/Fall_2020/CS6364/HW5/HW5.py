# Made by Daniel Crawford
# Student Net ID: dsc160130
# Course: CS6364 - Artificial Intelligence

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pandas import read_csv, get_dummies

dir = './'
HOUSING_DATA_FILE = dir + 'HousingData.csv'
TITANIC_DATA_FILE = dir + 'titanic3.csv'

RANDOM_STATE = 1234
LEARNING_RATE = 0.00001


def mse(true, pred, n):
    return np.sum((true - pred) ** 2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rmse(true, pred):
    return np.sqrt(((true - pred) ** 2).mean())


def rmse_derivative(true, pred, input_column, n):
    num = -1 * np.sum((true - pred) * input_column)
    den = (n * rmse(true, pred))
    if den == 0:
        return 0
    return num / den

def reportClassifierPerformance(y_train_true, y_train_pred, y_test_true, y_test_pred, name=''):
    row_format = "{:>25}" * 3
    num_decimals = 2
    scores = [[accuracy_score(y_train_true, y_train_pred) * 100, accuracy_score(y_test_true, y_test_pred) * 100],
              [precision_score(y_train_true, y_train_pred) * 100, precision_score(y_test_true, y_test_pred) * 100],
              [recall_score(y_train_true, y_train_pred) * 100, recall_score(y_test_true, y_test_pred) * 100],
              [f1_score(y_train_true, y_train_pred) * 100, f1_score(y_test_true, y_test_pred) * 100]]
    metric = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    print(row_format.format('', *['Train', 'Test']))
    for row in range(len(scores)):
        print(row_format.format(*[metric[row],
                                  str(round(scores[row][0], num_decimals)),
                                  str(round(scores[row][1], num_decimals))]
                                )
              )


class LinearClassifier:
    def __init__(self, lr, init_method='uniform', gd_method='standard'):
        self.lr = lr
        self.init_method = init_method
        self.derivative = None
        if gd_method == 'stochastic':
            self.gd_method = self.SGD
        elif gd_method == 'momentum':
            self.gd_method = self.MSGD
        elif gd_method == 'nestorov':
            self.gd_method = self.NMSGD
        elif gd_method == 'ada':
            self.gd_method = self.AdaGrad
        else:
            self.gd_method = self.GD

    def init_weights(self, num_features):
        self.num_features = num_features
        if self.init_method == 'uniform':
            return np.random.uniform(0, 1, num_features + 1)
        if self.init_method == 'normal':
            return np.random.normal(0, 1, num_features + 1)
        else:
            return np.zeros(num_features + 1)

    def train(self, train_data, train_target, epochs=20):
        train_data = np.array(train_data)
        train_target = np.array(train_target)
        self.weights = self.init_weights(len(train_data[0]))
        self.gd_method(train_data, train_target, epochs)

    def GD(self, train_data, train_target, epochs):
        n = len(train_data)
        for epoch in range(epochs):
            derivatives = np.array([self.derivative(train_target, self.predict(train_data), 1, n)] + \
                                   [self.derivative(train_target, self.predict(train_data), col, n) for col in
                                    train_data.transpose()])
            self.weights -= self.lr * derivatives

    def SGD(self, train_data, train_target, epochs):
        n = len(train_data)
        for epoch in range(epochs):
            for d, target in zip(train_data, train_target):
                derivative = np.array([self.derivative(target, self.predict(d), 1, 1)] + \
                                      [self.derivative(target, self.predict(d), col, 1) for col in d])
                self.weights -= self.lr * derivative

    def MSGD(self, train_data, train_target, epochs):
        n = len(train_data)
        nf = 0.9
        v = 0
        for epoch in range(epochs):
            derivatives = np.array([self.derivative(train_target, self.predict(train_data), 1, n)] + \
                                   [self.derivative(train_target, self.predict(train_data), col, n) for col in
                                    train_data.transpose()])
            v = nf * v - self.lr * derivatives
            self.weights += v

    def NMSGD(self, train_data, train_target, epochs):
        n = len(train_data)
        nf = 0.9
        v = np.zeros(len(self.weights))
        for epoch in range(epochs):
            derivatives = np.array([self.derivative(train_target, self.predict(train_data + n*v[0]), 1, n)] + \
                                   [self.derivative(train_target, self.predict(train_data + n*v[1:]), col, n) for col in
                                    train_data.transpose()])
            v = nf * v - self.lr * derivatives
            self.weights += v

    def AdaGrad(self, train_data, train_target, epochs):
        n = len(train_data)
        d = 1e-8
        r = 0
        for epoch in range(epochs):
            derivatives = np.array([self.derivative(train_target, self.predict(train_data), 1, n)] + \
                                   [self.derivative(train_target, self.predict(train_data), col, n) for col in
                                    train_data.transpose()])
            r = r + derivatives**2
            v = -1 * (self.lr / np.sqrt(d + r)) * derivatives
            self.weights += v


    def predict(self, data):
        if len(data.shape) > 1:
            return self.weights[0] + np.sum(self.weights[1:] * data, axis=1)
        return self.weights[0] + self.weights[1:].dot(data)


class LinearRegression(LinearClassifier):
    def __init__(self, lr, init_method='uniform', gd_method='standard'):
        super().__init__(lr, init_method, gd_method)
        self.derivative = rmse_derivative

    def init_weights(self, num_features):
        return super().init_weights(num_features)

    def train(self, train_data, train_target, epochs=20):
        super().train(train_data, train_target, epochs)

class LogisticRegression(LinearClassifier):
    def __init__(self, lr, init_method='uniform', gd_method='standard'):
        super().__init__(lr, init_method, gd_method)
        self.derivative = rmse_derivative

    def init_weights(self, num_features):
        return super().init_weights(num_features)

    def train(self, train_data, train_target, epochs=20):
        super().train(train_data, train_target, epochs)

    def predict(self, data):
        return super().predict(data)

    def predict_sigmoid(self, data):
        return np.round(sigmoid(super().predict(data)))


np.random.seed(RANDOM_STATE)


def q1(train_file):
    df = read_csv(train_file)
    target = 'MEDV'
    features = [col for col in df.columns if col != target]
    for col, type in zip(df.columns, df.dtypes):
        if type == 'str' or type == 'object':
            df[col] = df[col].fillna('None')
            df = df.join(get_dummies(df[col], prefix=col))
            df = df.drop(columns=col)
        else:
            df[col] = df[col].fillna(0)
    XTrain, Xtest, yTrain, ytest = train_test_split(df[features], df[target], test_size=0.3, random_state=RANDOM_STATE)
    model = LinearRegression(lr=LEARNING_RATE, init_method='zero', gd_method='standard')
    model.train(XTrain, yTrain, epochs=1000)
    print('-------------------------------')
    print('Standard Gradient Descent')
    print('-------------------------------')
    print('RMSE train: ', rmse(yTrain, model.predict(XTrain)))
    print('RMSE test: ', rmse(ytest, model.predict(Xtest)))
    print('-------------------------------')
    print('-------------------------------')
    print('Stochastic Gradient Descent')
    print('-------------------------------')
    model = LinearRegression(lr=LEARNING_RATE, init_method='zero', gd_method='stochastic')
    model.train(XTrain, yTrain, epochs=100)
    print('RMSE train: ', rmse(yTrain, model.predict(XTrain)))
    print('RMSE test: ', rmse(ytest, model.predict(Xtest)))
    print('-------------------------------')
    print('-------------------------------')
    print('Gradient Descent with momentum')
    print('-------------------------------')
    model = LinearRegression(lr=LEARNING_RATE, init_method='zero', gd_method='momentum')
    model.train(XTrain, yTrain, epochs=1000)
    print('RMSE train: ', rmse(yTrain, model.predict(XTrain)))
    print('RMSE test: ', rmse(ytest, model.predict(Xtest)))
    print('-------------------------------')
    print('-------------------------------')
    print('Gradient Descent with Nestorov momentum')
    print('-------------------------------')
    model = LinearRegression(lr=LEARNING_RATE, init_method='zero', gd_method='nestorov')
    model.train(XTrain, yTrain, epochs=1000)
    print('RMSE train: ', rmse(yTrain, model.predict(XTrain)))
    print('RMSE test: ', rmse(ytest, model.predict(Xtest)))
    print('-------------------------------')
    print('-------------------------------')
    print('Adagrad')
    print('-------------------------------')
    model = LinearRegression(lr=0.1, init_method='zero', gd_method='ada')
    model.train(XTrain, yTrain, epochs=1000)
    print('RMSE train: ', rmse(yTrain, model.predict(XTrain)))
    print('RMSE test: ', rmse(ytest, model.predict(Xtest)))
    print('-------------------------------')

def q2(train_file):
    df = read_csv(train_file).drop(columns=['name', 'ticket', 'fare', 'cabin', 'home.dest', 'body'])
    target = 'survived'
    for col, type in zip(df.columns, df.dtypes):
        if type == 'str' or type == 'object':
            df[col] = df[col].fillna('None')
            df = df.join(get_dummies(df[col], prefix=col))
            df = df.drop(columns=col)
        else:
            df[col] = df[col].fillna(0)
    features = [col for col in df.columns if col != target]
    XTrain, Xtest, yTrain, ytest = train_test_split(df[features], df[target], test_size=0.2, random_state=RANDOM_STATE)
    print('-------------------------------')
    print('Standard Gradient Descent')
    print('-------------------------------')
    model = LogisticRegression(lr=LEARNING_RATE, init_method='zero', gd_method='standard')
    model.train(XTrain, yTrain, epochs=1000)
    reportClassifierPerformance(yTrain, model.predict_sigmoid(XTrain), ytest, model.predict_sigmoid(Xtest), name='')
    print('-------------------------------')
    print('-------------------------------')
    print('Stochastic Gradient Descent')
    print('-------------------------------')
    model = LogisticRegression(lr=LEARNING_RATE, init_method='zero', gd_method='stochastic')
    model.train(XTrain, yTrain, epochs=10)
    reportClassifierPerformance(yTrain, model.predict_sigmoid(XTrain), ytest, model.predict_sigmoid(Xtest), name='')
    print('-------------------------------')
    print('-------------------------------')
    print('Gradient Descent with momentum')
    print('-------------------------------')
    model = LogisticRegression(lr=LEARNING_RATE, init_method='zero', gd_method='momentum')
    model.train(XTrain, yTrain, epochs=1000)
    reportClassifierPerformance(yTrain, model.predict_sigmoid(XTrain), ytest, model.predict_sigmoid(Xtest), name='')
    print('-------------------------------')
    print('-------------------------------')
    print('Gradient Descent with Nestorov momentum')
    print('-------------------------------')
    model = LogisticRegression(lr=LEARNING_RATE, init_method='zero', gd_method='nestorov')
    model.train(XTrain, yTrain, epochs=1000)
    reportClassifierPerformance(yTrain, model.predict_sigmoid(XTrain), ytest, model.predict_sigmoid(Xtest), name='')
    print('-------------------------------')
    print('-------------------------------')
    print('Adagrad')
    print('-------------------------------')
    model = LogisticRegression(lr=LEARNING_RATE, init_method='zero', gd_method='ada')
    model.train(XTrain, yTrain, epochs=1000)
    reportClassifierPerformance(yTrain, model.predict_sigmoid(XTrain), ytest, model.predict_sigmoid(Xtest), name='')
    print('-------------------------------')


q1(HOUSING_DATA_FILE)
q2(TITANIC_DATA_FILE)
