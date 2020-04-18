import xml.etree.ElementTree as ET
import re
from collections import defaultdict
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

n_epochs = 10
lstm_layers = 1
max_length = 25
batch_size = 5
train_split = 0.8
test_split = 0.2
hidden_dim = 50
embedding_dim = 50
lr = 1e-3
display_epoch = True



def reportPerformance(y_true, y_pred, description):
    print(description)
    print('Accuracy:  ', accuracy_score(y_true, y_pred), '\n',
          'Precision: ', precision_score(y_true, y_pred), '\n',
          'Recall:    ', recall_score(y_true, y_pred), '\n',
          'F-Score:   ', f1_score(y_true, y_pred), '\n', sep='')


trainFile = 'train.xml'
testFile = 'test.xml'


class Dataset(TensorDataset):
    def __init__(self, p=None, h=None, l=None):
        if p is None and h is None and l is None:
            return
        self.p = torch.from_numpy(np.array(p))
        self.h = torch.from_numpy(np.array(h))
        labels = np.array([[1 if i == 1 else 0, 1 if i == 2 else 0] for i in l])
        self.labels = torch.from_numpy(labels).type(torch.FloatTensor)

    def to(self, device):
        newDataset = Dataset()
        newDataset.p = self.p.to(device)
        newDataset.h = self.h.to(device)
        newDataset.labels = self.labels.to(device)
        return newDataset

    def __len__(self):
        return len(self.p)

    def __getitem__(self, item):
        n = len(self.p[item])
        return torch.cat((self.p[item].view(-1, n), self.h[item].view(-1, n))), self.labels[item]


class RNN(nn.Module):
    def __init__(self, vocab_length, embedding_dim, hidden_dim, output_dim, lstm_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_length, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.lstmL = []
        current_dim = hidden_dim * 2
        for i in range(lstm_layers):
            self.lstmL.append(nn.LSTM(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.lstmL = nn.ModuleList(self.lstmL)
        self.fc = nn.Linear(current_dim, output_dim)

    def forward(self, p, h):
        n = embedding_dim
        et = self.embedding(p)
        ht = self.embedding(h)
        rt, _ = self.lstm(et.view(n, 1, -1))
        rh, _ = self.lstm(ht.view(n, 1, -1))
        for lstm in self.lstmL:
            rt, _ = lstm(rt)
            rh, _ = lstm(rh)
        rth = torch.cat((rt.view(n, -1), rh.view(n, -1)), dim=0)
        fc = self.fc(torch.sum(rth, dim=0))
        return F.softmax(fc, dim=0)


class ModelTools():
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.throughput = 0

    def train(self, trainData, testData, N_EPOCHS=10, display_epoch=True):
        N_EPOCHS = N_EPOCHS
        throughput = []
        best_valid_loss = float('inf')
        for epoch in range(N_EPOCHS):
            t = time.time()
            train_loss, train_acc = self.trainModel(trainData)
            valid_loss, valid_acc = self.evaluateModel(testData)
            epoch_secs = time.time() - t
            throughput.append(epoch_secs)
            if valid_loss < best_valid_loss:
                torch.save(self.model.state_dict(), 'best_model.h1')
                best_valid_loss = valid_loss
            if display_epoch:
                print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_secs:.2f}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        self.throughput = sum(throughput) / len(throughput)

    def trainModel(self, data):

        epoch_loss = 0
        epoch_acc = 0

        self.model.train()

        for d in data:
            self.optimizer.zero_grad()
            p = [i[0] for i in d[0]]
            h = [i[1] for i in d[0]]
            batch_labels = torch.cat([i.unsqueeze(0) for i in d[1]])
            predictions = torch.cat([self.model(p[i], h[i]).unsqueeze(0) for i in range(len(p))])
            loss = self.criterion(predictions, batch_labels)
            acc = self.accuracy(predictions, batch_labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(data), epoch_acc / len(data)

    def evaluateModel(self, data):
        epoch_loss = 0
        epoch_acc = 0

        self.model.eval()

        with torch.no_grad():
            for d in data:
                p = [i[0] for i in d[0]]
                h = [i[1] for i in d[0]]
                batch_labels = torch.cat([i.unsqueeze(0) for i in d[1]])
                predictions = torch.cat([self.model(p[i], h[i]).unsqueeze(0) for i in range(len(p))])
                loss = self.criterion(predictions, batch_labels)
                acc = self.accuracy(batch_labels, predictions)
                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(data), epoch_acc / len(data)

    def predict(self, data):
        self.model.eval()
        predictions = torch.Tensor().type(torch.LongTensor).to(self.device)
        with torch.no_grad():
            for d in data:
                p = [i[0] for i in d[0]]
                h = [i[1] for i in d[0]]
                output = torch.cat([self.model(p[i], h[i]).argmax().view(1, -1) + 1 for i in range(len(p))])
                predictions = torch.cat((predictions, output))
        return torch.flatten(predictions).cpu().detach()

    def accuracy(self, y_true, y_pred):
        a = y_true.argmax(dim=1)
        b = y_pred.argmax(dim=1)
        return torch.sum(torch.eq(a, b)).type(torch.FloatTensor) / len(a)


class XMLToTensor:
    def __init__(self, files, max_length=None):
        data = np.array([self.parseXML(file) for file in files])
        self.p = np.array([p for f in data for p in f[0]])
        self.h = np.array([p for f in data for p in f[1]])
        self.l = np.array([p for f in data for p in f[2]])
        self.wd = self.createWordToIntegerDict([word for words in self.p for word in words] +
                                               [word for words in self.h for word in words])
        self.ld = self.createWordToIntegerDict([label for label in self.l])
        if max_length is None:
            self.max_length = max([len(s) for s in self.p + self.h])
        else:
            self.max_length = max_length

    def getTensor(self):
        p, h, l = self.prepareDataset(self.p, self.h, self.l)
        return Dataset(p, h, l)

    def splitTensor(self, trainSplit, testSplit):
        if trainSplit + testSplit != 1:
            print('Invalid split')
            return None
        n = len(self.p)
        p, h, l = self.prepareDataset(self.p, self.h, self.l)
        splitI = int(trainSplit * n)
        return Dataset(p[0:splitI], h[0:splitI], l[0:splitI]), Dataset(p[splitI:n], h[splitI:n], l[splitI:n])

    def getWordDictionary(self):
        return self.wd

    def getLabelDictionary(self):
        return self.ld

    def parseXML(self, file):
        tree = ET.parse(file)
        root = tree.getroot()
        pairs = root.findall('./pair')
        p = np.array([re.sub("[^\w]", " ", pair.find('./t').text.lower()).split() for pair in pairs])
        h = np.array([re.sub("[^\w]", " ", pair.find('./h').text.lower()).split() for pair in pairs])
        l = np.array([pair.get('value') for pair in pairs])
        return p, h, l

    def createWordToIntegerDict(self, words):
        d = defaultdict(int)
        i = 0
        for word in words:
            if d[word] == 0:
                i += 1
                d[word] = i
        return d

    def transformWordsToIntegers(self, w, d):
        return [[d[word] for word in words] for words in w]

    def transformListsToUniformLength(self, w, padding=0):
        L = [[padding for _ in range(self.max_length)] for _ in w]
        for i, l in enumerate(w):
            end = len(l) if len(l) <= self.max_length else self.max_length
            L[i][0:end] = l[0:end]
        return L

    def prepareDataset(self, p, h, l):
        p = self.transformListsToUniformLength(self.transformWordsToIntegers(p, self.wd))
        h = self.transformListsToUniformLength(self.transformWordsToIntegers(h, self.wd))
        l = [self.ld[l] for l in l]
        return p, h, l

def run():
    data = XMLToTensor([trainFile, testFile], max_length=max_length)
    train, test = data.splitTensor(train_split, test_split)
    wordDict = data.getWordDictionary()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNN(len(wordDict) + 1, embedding_dim, hidden_dim, 2, lstm_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model = model.to(device)
    criterion = criterion.to(device)
    train = train.to(device)
    test = test.to(device)
    trainSampler = RandomSampler(train)
    testSampler = SequentialSampler(test)
    train = DataLoader(train, batch_size=batch_size, sampler=trainSampler)
    test = DataLoader(test, batch_size=batch_size, sampler=testSampler)

    trainer = ModelTools(model, optimizer, criterion, device)
    trainer.train(train, test, N_EPOCHS=n_epochs, display_epoch=display_epoch)
    model = RNN(len(wordDict) + 1, embedding_dim, hidden_dim, 2, lstm_layers).to(device)
    model.load_state_dict(torch.load('best_model.h1'))
    y_pred = ModelTools(model, optimizer, criterion, device).predict(test).numpy()
    y_true = np.array([1 if i[0] == 1 else 2 for i in test.dataset.labels])

    print("Throughput: ", trainer.throughput)
    description = "Model with following specifications: \n" + \
                  "Number of epochs : " + str(n_epochs) + "\n" + \
                  "ExtraLSTM layers : " + str(lstm_layers) + "\n" + \
                  "Max word length  : " + str(max_length) + "\n" + \
                  "Batch size       : " + str(batch_size) + "\n" + \
                  "Train/Test split : " + str(train_split) + "/" + str(test_split) + "\n" + \
                  "Hidden dimension : " + str(hidden_dim) + "\n" + \
                  "Learning Rate    : " + str(lr) + "\n"
    reportPerformance(y_true, y_pred, description)