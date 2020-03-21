from os import listdir
from sys import argv
from os.path import exists
import argparse


def reportPerformance(y_pred, test_labels, description):
    print(description)
    print('Accuracy:  ', accuracy_score(test_labels, y_pred), '\n',
          'Precision: ', precision_score(test_labels, y_pred), '\n',
          'Recall:    ', recall_score(test_labels, y_pred), '\n',
          'F-Score:   ', f1_score(test_labels, y_pred), '\n', sep='')


parser = argparse.ArgumentParser(description='Arguments for training a basic word model')
parser.add_argument('train_dir', metavar='train_Dir', type=str,
                    help='Directory to training text files')
parser.add_argument('test_dir', metavar='test_dir', type=str,
                    help='Directory to testing text files')
parser.add_argument('representation', metavar='representation', type=str, choices=['bow', 'tfidf'],
                    help='Can either be \'bow\' or \'tfidf\'')
parser.add_argument('classifier', metavar='classifier', type=str, choices=['regression', 'nbayes'],
                    help='Can either be \'regression\' or \'nbayes\'')
parser.add_argument('stop_words', metavar='stop_words', type=int, choices=[0, 1],
                    help='Can either be \'0\' or \'1\'')
parser.add_argument('--regularization', '-r', metavar='reg', type=str, default=None,
                    choices=['no', 'l1', 'l2', ''], help='Can either be \'no\', \'l1\', \'l2\'')
args = parser.parse_args()
train_dir = args.train_dir + '/'
test_dir = args.test_dir + '/'
representation = args.representation
classifier = args.classifier
stop_words = args.stop_words
regularization = args.regularization
if not exists(train_dir):
    print('Error: train directory \'', train_dir,
          '\' does not exist.', sep='')
    exit(1)
if not exists(test_dir):
    print('Error: test directory \'', test_dir,
          '\' does not exist.', sep='')
    exit(1)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

stop_words = 'english' if stop_words == 1 else None
C = 9e7 if regularization == 'no' else 1.0
reg = 'l2' if regularization in ['no', None] else regularization
train_labels = []
train_files = []
test_labels = []
test_files = []
label_dir = ['neg', 'pos']

X = CountVectorizer(input='filename', stop_words=stop_words) if representation == 'bow' \
    else TfidfVectorizer(input='filename', stop_words=stop_words)
for l in label_dir:
    for file in listdir(train_dir + l + '/'):
        train_labels.append(0 if l == 'neg' else 1)
        train_files.append(train_dir + l + '/' + file)
    for file in listdir(test_dir + l + '/'):
        test_labels.append(0 if l == 'neg' else 1)
        test_files.append(test_dir + l + '/' + file)
X_train = X.fit_transform(train_files)
X_test = X.transform(test_files)
Model = LogisticRegression(penalty=reg, C=C, max_iter=1000, solver='liblinear') \
    if classifier == 'regression' else MultinomialNB()
F = Model.fit(X_train, train_labels)

classifierName = 'Logistic Regression' if classifier == 'regression' else 'Naive Bayes'
representationName = 'CountVectorizer' if representation == 'bow' else 'TfidfVectorizer'
stopWords = 'with stop words' if stop_words == 'english' else 'without stop words'
regWords = '' if regularization in ['no', None] else 'with ' + reg + ' regularization'
reportPerformance(F.predict(X_test), test_labels, classifierName + ' with ' + representationName + ' ' \
                  + stopWords + ' ' + regWords)