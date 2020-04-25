import pandas as pd
import numpy as np
import nltk
import pickle
import warnings
import sys
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=FutureWarning)


class Data_Extractor_NER:
    word_filler = 'UNK_WORD'
    pos_filler = 'UNK_TAG'
    pos_set = set()
    words_set = set()

    def __init__(self, file_name):
        df = self.parse_file(file_name)
        self.df = df
        self.num_sentences = df['Sentence #'].max()
        self.num_unique_tokens = len(df['Word'].unique())
        self.num_unique_tags = len(df['Tag'].unique())
        df['Sentence #'] = df['Sentence #'].astype(np.int32)
        self.int_token_dict = {}
        i = 1
        for e in df['Tag'].unique():
            self.int_token_dict[e] = i
            i += 1
        df['Tag'] = df['Tag'].apply(lambda tag: self.int_token_dict[tag])
        df['POS'] = self.get_pos_column(df, self.num_sentences)
        df['Word'] = self.lemmatize(df['Word'])
        words = list(df['Word'].unique()) + [self.word_filler]
        self.words_set = set(words)
        tags = list(df['POS'].unique()) + [self.pos_filler]
        self.pos_set = set(tags)
        self.ohe_word = OneHotEncoder()
        self.ohe_word.fit([[word] for word in words])
        self.ohe_pos = OneHotEncoder()
        self.ohe_pos.fit([[tag] for tag in tags])

    def get_test_matrix(self, test_file_name):
        df = self.parse_file(test_file_name)
        df['Tag'] = df['Tag'].apply(lambda tag: self.int_token_dict[tag])
        df['POS'] = self.get_pos_column(df, df['Sentence #'].max())
        df['Word'] = self.lemmatize(df['Word'])
        df['Word'] = df['Word'].apply(lambda word: word if word in self.words_set else self.word_filler)
        df['POS'] = df['POS'].apply(lambda pos: pos if pos in self.pos_set else self.pos_filler)
        return self.dataframe_to_matrix(df)

    def dataframe_to_matrix(self, df):
        labels = df['Tag'].to_numpy()
        sp1 = self.ohe_pos.transform([[pos] for pos in df['POS']])
        sp2 = self.ohe_word.transform([[word] for word in df['Word']])
        return hstack([sp1, sp2]).tocsr(), labels

    def get_train_matrix(self):
        return self.dataframe_to_matrix(self.df)

    def get_counts(self, df):
        num_sentences = df['Sentence #'].max()
        num_unique_tokens = len(df['Word'].unique())
        num_unique_tags = len(df['Tag'].unique())
        return num_sentences, num_unique_tokens, num_unique_tags

    def lemmatize(self, words):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return np.array([lemmatizer.lemmatize(word) for word in words])

    def parse_file(self, file_name):
        df = []
        sentence_no = 0
        new_sentence = True
        for i, line in enumerate(open(file_name).read().split('\n')):
            sentence_no_i = 0
            if i == 0:
                continue
            if line == '':
                sentence_no += 1
                new_sentence = True
                continue
            elif new_sentence:
                sentence_no_i = sentence_no
                new_sentence = False
            word, tag = line.split('\t')
            if word == '-DOCSTART-':
                sentence_no -= 1
                continue
            df.append([sentence_no_i] + [word, tag])
        df = pd.DataFrame(df, columns=['Sentence #', 'Word', 'Tag'])
        df['Word'] = df['Word'].str.lower()
        return df

    def get_inverse_tag_dict(self):
        return {v: k for k, v in self.int_token_dict.items()}

    @staticmethod
    def get_pos_column(df, num_sentences):
        words = df['Word'].to_numpy()
        sentence_no = df['Sentence #'].to_numpy()
        n = len(df)
        i = 0
        sentences = []
        for j in range(num_sentences):
            sentence = [words[i]]
            i += 1
            while i < n and sentence_no[i] == 0:
                sentence += [words[i]]
                i += 1
            sentences += [sentence]
        pos_tag = [nltk.pos_tag(sentence) for sentence in sentences]
        return [tag[1] for tags in pos_tag for tag in tags]


data_constructor = Data_Extractor_NER('modified_train.txt')
X_train, y_train = data_constructor.get_train_matrix()
X_test, y_test = data_constructor.get_test_matrix('modified_test.txt')
inverse_dict = data_constructor.get_inverse_tag_dict()

perceptron = Perceptron()
sgd = SGDClassifier()
pac = PassiveAggressiveClassifier()
nb = MultinomialNB()

model1 = perceptron.fit(X_train, y_train)
model2 = sgd.fit(X_train, y_train)
model3 = pac.fit(X_train, y_train)
model4 = nb.fit(X_train, y_train)


def report_results(model, X_train, y_test, inverse_dict, description):
    print(description)
    p = pickle.dumps(X_train)
    throughput = sys.getsizeof(p)
    t = time.time()
    y_pred = model.predict(X_train)
    t = time.time() - t
    throughput = throughput / t
    y_test = [inverse_dict[i] for i in y_test]
    y_pred = [inverse_dict[i] for i in model1.predict(X_test)]
    print(classification_report(y_test, y_pred))
    print('Accuracy: ', "{:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print('Throughput: ', "{:.2f}".format(throughput / 1024), 'KB/s')
    print('Inference time: ', "{:.6f}".format(t), 'seconds')
    print('\n\n')


# report_results(model1, X_test, y_test, inverse_dict, 'PERCEPTRON RESULTS: ')
# report_results(model2, X_test, y_test, inverse_dict, 'SGD CLASSIFIER RESULTS: ')
report_results(model3, X_test, y_test, inverse_dict, 'PASSIVE AGGRESSIVE CLASSIFIER RESULTS: ')
# report_results(model4, X_test, y_test, inverse_dict, 'MULTINOMIAL BAYES RESULTS: ')
