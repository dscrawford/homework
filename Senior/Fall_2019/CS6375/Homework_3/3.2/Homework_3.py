#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sklearn
import numpy as np

# In[2]:


from sklearn.datasets import fetch_openml
# Cross validation will be used so test set is not always defined.
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

Xtrain, ytrain = X[:60000], y[:60000]
Xtest, ytest = X[:60000], y[:60000]

# In[4]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
parameters = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': [1, 5, 10]}
svc = SVC(gamma = 'scale', cache_size = 1024, max_iter = 500, shrinking = True)
clf = GridSearchCV(estimator = svc, param_grid = parameters, n_jobs = 8)
clf.fit(X, y)


# In[5]:

for i,param in enumerate(clf.cv_results_['params']):
    print str(param) + str(': ') + str(1 - clf.cv_results_['mean_test_score'][i])


# In[6]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
parameters = {'activation': ['identity', 'logistic'], 
              'hidden_layer_sizes': [50, 100, 150],
              'solver': ['lbfgs', 'sgd', 'adam']}
mlp  = MLPClassifier()
clf2 = GridSearchCV(estimator = mlp, param_grid = parameters, n_jobs=8)
clf2.fit(X, y)


# In[14]:


clf2.best_estimator_


# In[12]:


for i,param in enumerate(clf2.cv_results_['params']):
    if (param['alpha'] == 1):
        print str(param) + str(': ') + str(1 - clf2.cv_results_['mean_test_score'][i])


# In[8]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
gbc = GradientBoostingClassifier(loss='deviance',criterion='mse', verbose=1, max_depth=5, max_features = 'sqrt')
gbc.fit(Xtrain, ytrain)

gbc2 = GradientBoostingClassifier(loss='deviance',criterion='mse', verbose=1, max_depth=4, max_features = 'sqrt')
gbc2.fit(Xtrain, ytrain)

gbc3 = GradientBoostingClassifier(loss='deviance',criterion='mse', verbose=1, max_depth=3, max_features = 'sqrt')
gbc3.fit(Xtrain, ytrain)

gbc4 = GradientBoostingClassifier(loss='deviance',criterion='mse', verbose=1, max_depth=2,
                                  max_features = 'sqrt')
gbc4.fit(Xtrain, ytrain)

gbc5 = GradientBoostingClassifier(loss='deviance',criterion='mse', verbose=1, max_depth=1,
                                  max_features = 'sqrt')
gbc5.fit(Xtrain, ytrain)


# In[9]:


gbc6 = GradientBoostingClassifier(loss='deviance',criterion='friedman_mse', verbose=1, max_depth=5,
                                  max_features = 'sqrt')
gbc6.fit(Xtrain, ytrain)

gbc7 = GradientBoostingClassifier(loss='deviance',criterion='friedman_mse', verbose=1, max_depth=4,
                                  max_features = 'sqrt')
gbc7.fit(Xtrain, ytrain)

gbc8 = GradientBoostingClassifier(loss='deviance',criterion='friedman_mse', verbose=1, max_depth=3,
                                  max_features = 'sqrt')
gbc8.fit(Xtrain, ytrain)

gbc9 = GradientBoostingClassifier(loss='deviance',criterion='friedman_mse', verbose=1, max_depth=2,
                                  max_features = 'sqrt')
gbc9.fit(Xtrain, ytrain)

gbc10 = GradientBoostingClassifier(loss='deviance',criterion='friedman_mse', verbose=1, max_depth=5,
                                  learning_rate = 0.01, max_features = 'sqrt')
gbc10.fit(Xtrain, ytrain)


# In[10]:


print 'mse with max depth 5 error rate: ' + str(1 - gbc.score(Xtest,ytest))
print 'mse with max depth 4 error rate: ' + str(1 - gbc2.score(Xtest,ytest))
print 'mse with max depth 3 error rate: ' + str(1 - gbc3.score(Xtest,ytest))
print 'mae with max depth 2 error rate: ' + str(1 - gbc4.score(Xtest,ytest))
print 'mae with max depth 1 error rate: ' + str(1 - gbc5.score(Xtest,ytest))
print 'friedman_mse with max depth 5 error rate: ' + str(1 - gbc6.score(Xtest,ytest))
print 'friedman_mse with max depth 4 error rate: ' + str(1 - gbc7.score(Xtest,ytest))
print 'friedman_mse with max depth 3 error rate: ' + str(1 - gbc8.score(Xtest,ytest))
print 'friedman_mse with max depth 2 error rate: ' + str(1 - gbc9.score(Xtest,ytest))
print 'friedman_mse with max depth 5 and learning rate 0.01 error rate: ' + str(1 - gbc10.score(Xtest,ytest))

