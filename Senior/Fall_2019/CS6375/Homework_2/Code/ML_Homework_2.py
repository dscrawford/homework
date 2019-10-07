#!/usr/bin/env python
# coding: utf-8

# In[52]:


from __future__ import division
import pandas as pd
import numpy  as np
import os
import re
from collections import Counter
import time, sys
import math

# Code made by Brain Khuu: https://stackoverflow.com/questions/3160699/python-progress-bar
# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "\nHalt...\r\n"
    if progress >= 1:
        progress = 1
        status = "\nDone...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


# In[54]:


folder = ""
if (len(sys.argv) > 2):
    print "ERROR: Too many arguments"
    sys.exit(1)
folder = "" if len(sys.argv) == 1 else sys.argv[1]
if (folder != "" and folder[len(folder) -1] != '/'):
    folder += '/'


# In[37]:


directory = folder


# In[38]:


# Condense multiple whitespaces into one, grab only alphabetic words, convert to lowercase
# and return array of words.
def getAllWordsFromString(words):
    return re.sub('\s+', ' ',re.sub('[^a-zA-Z1-9]+', ' ', words)).strip().lower().split(" ")

# getCountOfWords
# Create a dictionary with the count of each word in a string.
def getCountOfWords(words, allUniqueWords):
    allUniqueWordsDict = { i : 0 for i in allUniqueWords }
    counts = Counter(getAllWordsFromString(words))
    counts = {k : v for k, v in dict(counts).items() if k in allUniqueWordsDict}
    return mergeTwoDicts(allUniqueWordsDict, counts)

#getBernoulliWords
# Create a dictionary that shows the existence of words as 1 or 0.
def getBernoulliWords(words, allUniqueWords):
    counts = getCountOfWords(words, allUniqueWords)
    #Transform counts into existense
    return { k : (0 if v == 0 else 1) for k , v in counts.items()}
        
def getCountOfWordsWithProgressBar(words, allUniqueWords, progress):
    progress = round(progress,3)
    update_progress(progress)
    return getCountOfWords(words, allUniqueWords)

def getBernoulliWithProgressBar(words, allUniqueWords, progress):
    progress = round(progress,3)
    update_progress(progress)
    return getBernoulliWords(words, allUniqueWords)

def mergeTwoDicts(x, y):
    z = x.copy()
    z.update(y)
    return z


# In[39]:


#getProduct of Probabilities
# returns the log probability sum of all of the elements based on bayes
# works with both bernoulli and bag of words model
def getProductOfProbabities(text, T):
    featureSums = T.sum().loc[[w for w in getAllWordsFromString(text) if w in T.columns]]
    totalWords  = T.sum().sum()
    return np.log( (featureSums + 1) / (totalWords + len(T.columns))).sum()

def naiveBayesOnModel(text, T):
    p_0 = np.log(len(T[T['isSpam'] == 0]) / len(T)) + getProductOfProbabities(text, (T[T['isSpam'] == 0]).drop('isSpam', axis=1))
    p_1 = np.log(len(T[T['isSpam'] == 1]) / len(T)) + getProductOfProbabities(text, (T[T['isSpam'] == 1]).drop('isSpam', axis=1))
    return 0 if p_0 > p_1 else 1
    
def getDirectoryContents(dataDirectory):
    contents = np.array([])
    for fileName in os.listdir(dataDirectory):
        contents = np.append(contents, [open(dataDirectory + fileName).read()])
    return contents

def getBagOfWordsDataFrame(data, allUniqueWords):
    print "Creating DataFrame with Bag Of Words as the feature..."
    attributes = set(allUniqueWords)
    df = pd.DataFrame([getCountOfWordsWithProgressBar(d[1], attributes, i / (len(data) - 1))
                       for i,d in enumerate(data)])
    df.insert(0, 'isSpam', [d[0] for d in data])
    return df

def getBernoulliDataFrame(data, allUniqueWords):
    print "Creating DataFrame with Bernoulli model as the feature..."
    attributes = set(allUniqueWords)
    df = pd.DataFrame([getBernoulliWithProgressBar(d[1], attributes, i / (len(data) - 1))
                       for i,d in enumerate(data)])
    df.insert(0, 'isSpam', [d[0] for d in data])
    return df

def getNaiveBayesPredictions(Test, Train):
    return Test.apply(lambda x: naiveBayesOnModel(x['text'], Train), axis=1)

def getAccuracyOnNaiveBayes(Test, Train):
    return sum(Test.apply(lambda x: naiveBayesOnModel(x['text'], Train) == x['isSpam'], axis=1)) / len(Test)    

def PredictWithLR(T, W):
    bias = W[0]
    PY_1 = 1 / (1 + math.exp(bias + 
                             T.apply(lambda x: (T[x].sum() / T.sum().sum()) * W[x]).sum()))
    PY_0 = 1 - PY_1
    
    return 1


# In[40]:


def getProbYIsZero(scores):
    return 1 / (1 + np.exp(-scores))

def getProbYIsOne(scores):
    return 1 - getProbYIsZero(scores)

def getWeight(W, T):
    predictions = getPredictions(W, T)
    target      = T['isSpam']
    attributes  = T.drop('isSpam', axis=1)
    attributes.insert(0, 'x_0', 1)
    gradient    = np.dot(attributes.T, target - predictions)
    return gradient.astype(np.float64)

def getLogLikelihood(W, T):
    target = T['isSpam']
    features = T.drop('isSpam', axis=1)
    features.insert(0, 'isSpam', 1)
    scores = np.dot(features, W)
    return np.sum(target * scores - np.log(1 + np.exp(-scores)))

def getPredictions(W, T):
    features = T.drop('isSpam', axis=1)
    features.insert(0, 'w_0', 1)
    return getProbYIsZero(np.dot(T,W)).astype(np.float64)
    
def getAccuracyOnLR(W, T):
    return np.sum([T['isSpam'][i] == prediction.round() for i, prediction in enumerate(getPredictions(W,T))]) / len(T)

def splitDataFrame(D, frac):
    return (D[0: int(math.floor(len(D) * frac))], D[int(math.floor(len(D) * frac)): len(D)])

def L2Regularization(W, V, penalty):
    target = V['isSpam']
    predictions = getPredictions(W, V)
    features = V.drop('isSpam', axis=1)
    features.insert(0,'x_0', 1)
    gradient = np.dot(features.T, target - predictions)
    return (gradient - penalty * W)
    
def logisticRegression(D, numSteps, learningRate, penalty):
    W = np.zeros(len(D.columns))
    ham1, ham2   = splitDataFrame(D[D['isSpam'] == 0], 0.7)
    spam1, spam2 = splitDataFrame(D[D['isSpam'] == 1], 0.7)
    T = ham1.append(spam1).reset_index(drop=True)
    V = ham2.append(spam2).reset_index(drop=True)
    
    print "Performing gradient descent with weights starting at 0"
    for i in range(1, numSteps):
        W += learningRate * getWeight(W,T)
        update_progress( round(i / (numSteps - 1), 3))
    print "Regularizing weights using L2 Regularization"
    for i in range(1, numSteps):
        W += learningRate * L2Regularization(W, V, penalty)
        update_progress( round(i / (numSteps - 1), 3))
    return W


# In[41]:


trainHamData  = [[0,f] for f in getDirectoryContents(directory + "train/ham/")]
trainSpamData = [[1,f] for f in getDirectoryContents(directory + "train/spam/")]
allTrainData  = trainHamData + trainSpamData
testHamData   = [[0,f] for f in getDirectoryContents(directory + "test/ham/")]
testSpamData  = [[1,f] for f in getDirectoryContents(directory + "test/spam/")]
allTestData   = pd.DataFrame(testHamData + testSpamData).rename(columns={0: 'isSpam', 1: 'text'})


# In[42]:


#Transform all files into a single string.
allTrainWords = ''.join([f[1] for f in allTrainData])
#Retrieve all unique WORDS - Remove all words with numbers/punctuation and replace with space.
allUniqueWords = np.unique(getAllWordsFromString(allTrainWords))


# In[43]:


#Get a dataframe with bernoulli as the feature
trainB = getBernoulliDataFrame(allTrainData, allUniqueWords)
testB  = getBernoulliDataFrame(testHamData + testSpamData, allUniqueWords)
#Get a dataframe with bag of words as a feature for training
trainBOW = getBagOfWordsDataFrame(allTrainData, allUniqueWords)
testBOW  = getBagOfWordsDataFrame(testHamData + testSpamData, allUniqueWords)


# In[44]:


# P(Y = 0 | X)
def getProbYIsZero(scores):
    return 1 / (1 + np.exp(-scores))

# P(Y = 1 | X)
def getProbYIsOne(scores):
    return 1 - getProbYIsZero(scores)

# Returns weight after gradient descent without learning rate.
def getWeight(W, T):
    predictions = getPredictions(W, T)
    target      = T['isSpam']
    attributes  = T.drop('isSpam', axis=1)
    attributes.insert(0, 'x_0', 1)
    gradient    = np.dot(attributes.T, target - predictions)
    return gradient.astype(np.float64)

# Return log likelihood on dataset
def getLogLikelihood(W, T):
    target = T['isSpam']
    features = T.drop('isSpam', axis=1)
    features.insert(0, 'isSpam', 1)
    scores = np.dot(features, W)
    return np.sum(target * scores - np.log(1 + np.exp(-scores)))

# Return all predictions before threshold
def getLRPredictions(W, T):
    features = T.drop('isSpam', axis=1)
    features.insert(0, 'w_0', 1)
    return getProbYIsZero(np.dot(T,W)).astype(np.float64)
    
# Returns 
def getAccuracyOnLR(W, T):
    return np.sum([T['isSpam'][i] == prediction.round() for i, prediction in enumerate(getPredictions(W,T))]) / len(T)

def splitDataFrame(D, frac):
    return (D[0: int(math.floor(len(D) * frac))], D[int(math.floor(len(D) * frac)): len(D)])

def L2Regularization(W, V, penalty):
    target = V['isSpam']
    predictions = getPredictions(W, V)
    features = V.drop('isSpam', axis=1)
    features.insert(0,'x_0', 1)
    gradient = np.dot(features.T, target - predictions)
    return (gradient - penalty * W)
    
def logisticRegression(D, numSteps, learningRate, penalty):
    W = np.zeros(len(D.columns))
    ham1, ham2   = splitDataFrame(D[D['isSpam'] == 0], 0.7)
    spam1, spam2 = splitDataFrame(D[D['isSpam'] == 1], 0.7)
    T = ham1.append(spam1).reset_index(drop=True)
    V = ham2.append(spam2).reset_index(drop=True)
    
    print "Performing gradient descent with weights starting at 0"
    for i in range(1, numSteps):
        W += learningRate * getWeight(W,T)
        update_progress( round(i / (numSteps - 1), 3))
    print "Regularizing weights using L2 Regularization"
    for i in range(1, numSteps):
        W += learningRate * L2Regularization(W, V, penalty)
        update_progress( round(i / (numSteps - 1), 3))
    return W


# In[45]:


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm, datasets

np.random.seed(12)

parameters = dict(penalty=['l1','l2', 'elasticnet'], 
                  loss=['squared_loss', 'huber', 'epsilon_insensitive', 'hinge', 'log',
                        'modified_huber', 'squared_hinge'])

print "Fitting SGDClassifier on Bag of Words features and finding best parameters using GridSearchCV..."
X = trainBOW.drop('isSpam', axis=1)
Y = trainBOW['isSpam']
clfBOW = SGDClassifier(max_iter=1000, tol=1e-3)
clfBOW = GridSearchCV(clfBOW, parameters, cv=5, verbose=0)
clfBOW.fit(X,Y)

print "Fitting SGDClassifier on Bernoulli features and finding best parameters using GridSearchCV..."
X = trainB.drop('isSpam', axis=1)
Y = trainB['isSpam']
clfBer = SGDClassifier(max_iter=1000, tol=1e-3)
clfBer = GridSearchCV(clfBer, parameters, verbose=0)
clfBer.fit(X,Y)


# In[46]:


def getAccPrecRecallF1(Test, predictions):
    positives = list(testBOW[testBOW['isSpam'] == 1]['isSpam'].index)
    negatives = list(testBOW[testBOW['isSpam'] == 0]['isSpam'].index)
    true_positives  = sum(predictions[positives] == testBOW['isSpam'][positives])
    false_positives = sum(predictions[positives] != testBOW['isSpam'][positives])
    true_negatives  = sum(predictions[negatives] == testBOW['isSpam'][negatives])
    false_negatives = sum(predictions[negatives] != testBOW['isSpam'][negatives])
    
    accuracy        = sum(predictions == testBOW['isSpam']) / len(predictions)
    precision       = true_positives / (true_positives + false_positives)
    recall          = true_positives / (true_positives + false_negatives)
    F1              = 2 * (precision * recall) / (precision + recall)
    
    return (accuracy, precision, recall, F1)

def printAccPrecRecallF1(Test, predictions, algorithmName, featureType):
    acc, prec, recall, F1 = getAccPrecRecallF1(Test, predictions)
    print "Performance with " + algorithmName + " using " + featureType + " as features:"
    print "Accuracy : " + str(acc)
    print "Precision: " + str(prec)
    print "Recall   : " + str(recall)
    print "F1       : " + str(F1)


# In[47]:


print "Finding weights for Bag of words with Logistic Regression..."
WBOW = logisticRegression(trainBOW, 1000, 0.001, 0.1)
print "Finding weights for Bernoulli with Logistic Regression..."
WBer = logisticRegression(trainB,   1000, 0.001, 0.1)


# In[48]:


print "Finding predictions for all models..."
multipredictions    = getNaiveBayesPredictions(Test=allTestData, Train=trainBOW)
discretepredictions = getNaiveBayesPredictions(Test=allTestData, Train=trainB)
LRBOWpredictions    = getLRPredictions(WBOW, testBOW).round()
LRBerpredictions    = getLRPredictions(WBer, testB).round()
SGDBOWpredictions   = clfBOW.predict(testBOW.drop('isSpam', axis=1))
SGDBerpredictions   = clfBer.predict(testB.drop('isSpam', axis=1))


# In[49]:


target = trainBOW
printAccPrecRecallF1(target, multipredictions, "Multinomial Bayes", "Bag of Words")
printAccPrecRecallF1(target, discretepredictions, "Discrete Bayes", "Bernoulli")
printAccPrecRecallF1(target, LRBOWpredictions, "Logistic Regression", "Bag of Words")
printAccPrecRecallF1(target, LRBerpredictions, "Logistic Regression", "Bernoulli")
printAccPrecRecallF1(target, SGDBOWpredictions, "SGDClassifier", "Bag of Words")
printAccPrecRecallF1(target, SGDBerpredictions, "SGDClassifier", "Bernoulli")

