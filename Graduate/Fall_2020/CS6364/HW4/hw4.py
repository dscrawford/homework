# Made by Daniel Crawford
# Student Net ID: dsc160130
# Course: CS6364 - Artificial Intelligence

import pandas as pd
import numpy as np
import warnings

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 1234

HOUSING_DATA_FILE = 'HousingData.csv'
TITANIC_FILE = 'titanic3.csv'
DIABETES_FILE = 'diabetes.csv'
ADVERTISING_FILE = 'advertising.csv'


def reportRMSE(y_true, y_pred, name):
    print(name + ' RMSE: ', np.sqrt(mean_squared_error(y_true, y_pred)))


def reportClassifierPerformance(y_train_true, y_train_pred, y_test_true, y_test_pred, name):
    print(name)
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


def question1():
    model = linear_model.LinearRegression()
    df = pd.read_csv(HOUSING_DATA_FILE)
    df = df.fillna(0)
    target = 'MEDV'
    features = [col for col in df.columns if col != target]
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    reportRMSE(y_train, model.predict(X_train), 'Housing train')
    reportRMSE(y_test, model.predict(X_test), 'Housing test')


def question2():
    model = linear_model.LinearRegression()
    df = pd.read_csv(ADVERTISING_FILE)
    df = df.fillna(df.mean())
    target = 'Sales'
    features = [col for col in df.columns if col != target]
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    reportRMSE(y_train, model.predict(X_train), 'Advertising train')
    reportRMSE(y_test, model.predict(X_test), 'Advertising test')


def question3():
    model = linear_model.LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    df = pd.read_csv(TITANIC_FILE).drop(columns=['name', 'ticket', 'fare', 'cabin', 'home.dest', 'body'])
    for col, type in zip(df.columns, df.dtypes):
        if type == 'str' or type == 'object':
            df[col] = df[col].fillna('None')
            df = df.join(pd.get_dummies(df[col], prefix=col))
            df = df.drop(columns=col)
        else:
            df[col] = df[col].fillna(0)
    target = 'survived'
    features = [col for col in df.columns if col != target]
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    reportClassifierPerformance(y_train, model.predict(X_train), y_test, model.predict(X_test), 'Titanic Performance')


def question4():
    model = linear_model.LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    df = pd.read_csv(DIABETES_FILE)
    target = 'Outcome'
    features = [col for col in df.columns if col != target]
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    reportClassifierPerformance(y_train, model.predict(X_train), y_test, model.predict(X_test), 'Diabetes Dataset')


questions = [question1, question2, question3, question4]
for i, f in enumerate(questions):
    print('QUESTION ' + str(i + 1))
    f()
    print('-----------------------------------------------------------------------------------\n\n')
