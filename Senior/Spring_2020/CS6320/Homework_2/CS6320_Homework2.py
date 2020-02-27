from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from os import listdir
from sys import argv
from os.path import exists


def reportPerformance(y_pred, test_labels, model_name):
    print(model_name)
    print('Accuracy:  ', accuracy_score(y_pred, test_labels), '\n',
          'Precision: ', precision_score(y_pred, test_labels), '\n',
          'Recall:    ', recall_score(y_pred, test_labels), '\n',
          'F-Score:   ', f1_score(y_pred, test_labels), '\n', sep='')


if len(argv) != 7:
    print('Error: Insufficient amount of arguments')
    exit(1)
train_dir = argv[1] + '/'
test_dir = argv[2] + '/'
representation = argv[3]
classifier = argv[4]
stop_words = argv[5]
regularization = argv[6]
if not exists(train_dir):
    print('Error: train directory \'', train_dir,
          '\' does not exist.', sep='')
    exit(1)
if not exists(test_dir):
    print('Error: test directory \'', test_dir,
          '\' does not exist.', sep='')
    exit(1)
if representation not in ['bow', 'tfidf']:
    print('Error: representation \'', representation,
          '\' is not supported', sep='')
    exit(1)
if classifier not in ['nbayes', 'regression']:
    print('Error: classifier \'', classifier,
          '\' is not supported', sep='')
    exit(1)
if stop_words not in ['0', '1']:
    print('Error: stop words must be 0 or 1')
    exit(1)
if regularization not in ['no', 'l1', 'l2']:
    print('Error: regularization must be no, l1 or l2')
    exit(1)
stop_words = 'english' if stop_words == '1' else ''
C = 9e7 if regularization == 'no' else 1.0
regularization = 'l2' if regularization == 'no' else regularization
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
Model = LogisticRegression(penalty=regularization, C=C, max_iter=1000) if classifier == 'regression' \
    else MultinomialNB()
F = Model.fit(X_train, train_labels)

classifierName = 'Logistic Regression' if classifier == 'regression' else 'Naive Bayes'
representationName = 'CountVectorizer' if representation == 'bow' else 'TfidfVectorizer'
reportPerformance(F.predict(X_test), test_labels, classifierName + ' with ' + representationName)
