#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
from __future__ import print_function
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
from collections import defaultdict
import pandas as pd
import numpy  as np
import random


# In[64]:


def CLT(T, P, r):
    n   = len(T[0])
    Tt  = np.transpose(T)
    if (type(P) != list):
        P = 1
    Iuv = np.array([[MI(Tt, u, v, P) for u in range(n)] for v in range(n)])
    
    # Option to remove arguments from 
    l = list(range(n*n))
    np.random.shuffle(l)
    new = np.ndarray.flatten(Iuv)
    new[l[0:r]] = 0
    Iuv = np.reshape(new, (n,n))
    return np.array(nx.adjacency_matrix(nx.maximum_spanning_tree(nx.from_numpy_array(Iuv))).todense())

def MI(T, u, v, P):
    if (u == v):
        return 0
    Iu = [T[u] == 0, T[u] == 1]
    Iv = [T[v] == 0, T[v] == 1]
    Pu = [(np.sum(Iu[u] * P) + 1) / (len(Iu[u] * P) + 2) for u in [0,1]]
    Pv = [(np.sum(Iv[v] * P) + 1) / (len(Iv[v] * P) + 2) for v in [0,1]]
    Puv = [[(np.sum(np.logical_and(Iu[u],Iv[v]) * P) + 1) / (np.sum(np.logical_or(Iu[u],Iv[v])) + 2) for v in [0,1]] for u in [0,1]]
    return np.sum([np.sum([Puv[u][v] * (np.log2(Puv[u][v]) - np.log2(Pu[u]) - np.log2(Pv[v])) for u in [0,1]]) for v in [0,1]])


# In[6]:


data_dir = "./small-10-datasets/"
data_titles = ['accidents', 'baudio', 'bnetflix', 'dna', 'jester', 'kdd', 'msnbc',
              'nltcs', 'plants', 'r52']
test  = dict()
train = dict()
valid = dict()

for title in data_titles:
    test[title]  = np.loadtxt(data_dir + title + '.test.data', delimiter=',')
    train[title] = np.loadtxt(data_dir + title + '.ts.data', delimiter=',')
    valid[title] = np.loadtxt(data_dir + title + '.valid.data', delimiter=',')


# In[7]:


# Log Likelihood
def LL(p):
    return np.sum(np.log2(p))

def AVG_LL(P):
    return sum([LL(p) for p in P]) / len(P)

# Log Sum Exponent
def lse(a):
    m = max(a)
    return np.log2(np.sum(np.power(2., a - m))) + m

def Split_Tree(T, k):
    split = [i for i in range(0, len(T), int(len(T) / k))]
    return [T[range(split[i],len(T) if i == k -1 else split[i+1])] for i in range(k)]

# Transforms MST into a DAG, and then finds associated probabilities as well.
def Create_Network(MST, T):
    n = len(MST)
    T = np.transpose(T)
    network = [{} for i in range(n)]
    root = 0

    p = (sum(T[root] == 1) + 1) / (len(T[root]) + 2)
    network[root] = {root : [1 - p, p]}
    children = np.ndarray.flatten(np.argwhere(np.transpose(MST[root]) != 0))
    parents = {root : children}
    for c in children:
        MST[c][root] = 0.

    while (len(parents) != 0):
        newParents   = {}
        for parent in parents:
            children = parents[parent]
            for c in children:
                # Remove edge, make directed
                p = [(sum(T[c][T[parent] == 0] == 1) + 1) / (sum(T[parent] == 0) + 2),
                     (sum(T[c][T[parent] == 1] == 1) + 1) / (sum(T[parent] == 1) + 2)]
                network[c].update({parent : [1 - p[0], p[0], 1 - p[1], p[1]]})
                cc = np.ndarray.flatten(np.argwhere(np.transpose(MST[c]) != 0))
                for child in cc:
                    MST[child][c] = 0
                if (len(children) != 0):
                    newParents.update({c : cc})
        parents = newParents    
    return network;

# Predicts a network generated from above.
def Predict_Network(N, test):
    all_predictions = []
    for t in test:
        predictions = []
        for i in range(len(N)):
            probs = []
            for k in N[i].keys():
                if(k == i):
                    probs.append(N[i][k][0] if t[i] == 0 else N[i][k][1])
                else:
                    if (t[k] == 0):
                        probs.append(N[i][k][0] if t[i] == 0 else N[i][k][1])
                    else:
                        probs.append(N[i][k][2] if t[i] == 0 else N[i][k][3])
            predictions.append(np.product(probs))
        all_predictions.append(predictions)
    return np.array(all_predictions)

def Predict_Mixture(M, test):
    N  = M[1]
    pi = M[0]
    n = len(pi)
    predictions = []
    for i in range(n):
        predictions.append(pi[i] * Predict_Network(N[i], test))
    return np.sum(predictions, axis=0)


# In[77]:


# Bayesian Network, No edges Algorithm
# Takes in a dataset of binary variables and takes a test set of binary variables.
def BN_NE(T, test):
    cols = np.transpose(T)
    n    = len(cols)
    p_1 = np.array([(sum(cols[i] == 1) + 1) / (len(cols[i]) + 2) for i in range(n)])
    return np.array([[p_1[+i] if ti == 0 else 1 - p_1[i] for i, ti in enumerate(t)] for t in test])

# Bayesian Network, Chow-Liu Algorithm
# Returns a data structure in the form of a Bayesian Network with probabilities pre-computed inside.
def BN_CL(T, P, r):
    n = len(T[0])
    tree = Create_Network(CLT(T, P, r), T)
    return tree

# Mixtures of Tree Bayesian networks using EM
def MT_BN(T, V, M):
    # Validation
    MTs = [EM(T, m, 100) for m in M]
    P   = [AVG_LL(Predict_Mixture(MT, V)) for MT in MTs]
    
    return MTs[np.argmax(P)]

# Mixtures of Tree Bayesian Networks using Random Forests
def MT_BN_RF(T, V, K, R):
    # Validation
    combs = [(k,r) for r in R for k in K]
    RFs   = [RF(T, k, r) for k,r in combs]
    P     = [AVG_LL(Predict_Mixture(rf, V)) for rf in RFs]
    
    return RFs[np.argmax(P)]
    
def RF(T, k, r):
    n = len(T)
    Ns = [BN_CL(T[np.random.uniform(0,n,n).astype(int)], 1, r) for k in range(k)]
    pi = [random.random() for i in range(k)]
    pi = [pi[i] / sum(pi) for i in range(k)]
    return [pi, Ns]
    

# Expectation Maximization
def EM(T, m, max_iter):
    split = Split_Tree(T, m)
    r = np.array([random.random() for i in range(m)]).astype(np.float64)
    pi = [i / sum(r) for i in r]
    yk = [[random.random() for j in range(m)] for i in range(len(T))]
    yk = np.array([[yk[i][j] / sum(yk[i]) for j in range(m)] for i in range(len(T))])
    N  = [BN_CL(T, list(y), 0) for y in np.transpose(yk)] 

    prevPi = []
    it = 0
    ll = AVG_LL(Predict_Mixture([pi,N], T))
    prev_ll = ll - 1
    while (prev_ll < ll and it != max_iter):
        prev_ll = ll
        it += 1
        
        # E-step
        W = []
        p = [[pi[k] * np.product(Predict_Network(N[k],[t])) for k in range(m)] for t in T]
        yk = np.array([[p[i][k] / sum(p[i]) for k in range(m)] for i in range(len(T))])
        rk = [sum(k) for k in np.transpose(yk)]
        pk = [[yk[i][k] / rk[k] for k in range(m)] for i in range(len(T))]
        
        # M-step
        pi = [rk[k] / sum(rk) for k in range(m)]
        N  = [BN_CL(T, list(y), 0) for y in np.transpose(yk)] 
        
        ll = AVG_LL(Predict_Mixture([pi,N], T))
    return [pi, N]


# In[73]:


NE_LL = {k : 0 for k in data_titles}
BN_CL_LL = {k : 0 for k in data_titles}
MT_BN_LL = {k : 0 for k in data_titles}
MT_BN_RF_LL = {k : 0 for k in data_titles}

for title in data_titles:
    NE_LL[title] = AVG_LL(BN_NE(train[title], test[title]))
    BN_CL_LL[title] = AVG_LL(Predict_Network(BN_CL(train[title], 1, 0), 
                                             test[title]))
    MT_BN_LL[title] = AVG_LL(Predict_Mixture(MT_BN(train[title], valid[title], 
                                                   [3,5,10]), test[title]))
    n = len(train[title])
    MT_BN_RF_LL[title] = AVG_LL(Predict_Mixture(MT_BN_RF(train[title], valid[title],
                                [3,5,10], [n // 3, n // 5, n // 10]), test[title]))


# In[ ]:


MT_BN_LL_2 = {k : 0 for k in data_titles}
for title in data_titles:
    print(title)
    MT_BN_LL_2[title] = AVG_LL(Predict_Mixture(MT_BN(train[title], valid[title], 
                                                   [3,5,10]), test[title]))


# In[76]:


print_str = ""
for title in data_titles:
    print_str += "No edge LL of set \'" + title + "\':" + str(NE_LL[title]) + "\n"
for title in data_titles:
    print_str += "Bayesian Network LL of set \'" + title + "\':" + str(BN_CL_LL[title]) + "\n"
for title in data_titles:
    print_str += "Mixture Tree EM LL of set \'" + title + "\':" + str(MT_BN_LL_2[title]) + "\n"
for title in data_titles:
    print_str += "Mixture Tree RF LL of set \'" + title + "\':" + str(MT_BN_RF_LL[title]) + "\n"
    
print(print_str)


# In[ ]:




