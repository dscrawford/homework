import torch
import torch.nn as nn
import numpy as np

from pandas import get_dummies, read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dir = './'
HOUSING_DATA_FILE = dir + 'HousingData.csv'
TITANIC_DATA_FILE = dir + 'titanic3.csv'
RANDOM_STATE = 1234

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def RMSELoss(y, y_true):
    return torch.sqrt(torch.mean((y_true - y) ** 2))


class RegressionNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2):
        super(RegressionNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size_1)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.t1 = nn.Tanh()
        self.l3 = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        out = self.l1(x)
        out = self.r1(out)
        out = self.l2(out)
        out = self.t1(out)
        out = self.l3(out)
        return out

class ClassifierNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2):
        super(ClassifierNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size_1)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.t1 = nn.Tanh()
        self.l3 = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        out = self.l1(x)
        out = self.r1(out)
        out = self.l2(out)
        out = self.t1(out)
        out = self.l3(out)
        return torch.sigmoid(out)


def train_model(X_train, y_train, model, criterion, optimizer, num_epochs=20):
    model.train()
    X_train = torch.from_numpy(np.array(X_train)).float().to(device)
    y_train = torch.from_numpy(np.array(y_train)).float().to(device)
    for epoch in range(num_epochs):
        train_loss = []
        for i, (features, true) in enumerate(zip(X_train, y_train)):
            # Forward pass
            outputs = model(features)[0]
            loss = criterion(outputs, true)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
        print('\rEpoch [{}/{}], Train Loss: {:.4f}'
              .format(epoch + 1, num_epochs, np.sqrt(np.mean(train_loss))), end='')
    print()

def eval_model_mse(X_test, y_test, model):
    model.eval()
    X_test = torch.from_numpy(np.array(X_test)).float().to(device)
    y_test = torch.from_numpy(np.array(y_test)).float().to(device)
    loss = []
    for features, true in zip(X_test, y_test):
        output = model(features)[0]
        loss.append((true.item() - output.item())**2)
    print('Test RMSE:', np.sqrt(np.mean(loss)))

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

def eval_model_classifier(X_test, y_test, model):
    model.eval()
    X_test = torch.from_numpy(np.array(X_test)).float().to(device)
    y_test = torch.from_numpy(np.array(y_test)).float().to(device)
    predictions = []
    for features, true in zip(X_test, y_test):
        output = model(features)[0]
        predictions.append(np.round(output.item()))
    return np.array(predictions)



def q1():
    df = read_csv(HOUSING_DATA_FILE)
    target = 'MEDV'
    for col, type in zip(df.columns, df.dtypes):
        if type == 'str' or type == 'object':
            df[col] = df[col].fillna('None')
            df = df.join(get_dummies(df[col], prefix=col))
            df = df.drop(columns=col)
        else:
            df[col] = df[col].fillna(0)
    features = [col for col in df.columns if col != target]

    XTrain, Xtest, yTrain, ytest = train_test_split(df[features], df[target], test_size=0.2, random_state=RANDOM_STATE)

    input_size = len(features)
    model = RegressionNN(input_size, 16, 32).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(XTrain, yTrain, model, criterion, optimizer, num_epochs=100)
    eval_model_mse(Xtest, ytest, model)

def q2():
    df = read_csv(TITANIC_DATA_FILE).drop(columns=['name', 'ticket', 'fare', 'cabin', 'home.dest', 'body'])
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

    input_size = len(features)
    model = ClassifierNN(input_size, 5, 3).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(XTrain, yTrain, model, criterion, optimizer, num_epochs=100)
    reportClassifierPerformance(yTrain,
                                eval_model_classifier(XTrain, yTrain, model),
                                ytest,
                                eval_model_classifier(Xtest, ytest, model))

print('Boston Housing Dataset: ')
q1()
print()
print('Titanic Datset: ')
q2()
