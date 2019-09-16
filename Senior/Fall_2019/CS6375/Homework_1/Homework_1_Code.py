#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import math
import copy
import sys
from itertools import chain
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[2]:


#lg(P)
#  return log_2(P) if P != 0, else return 0
def lg(P):
    return math.log(P, 2)

def Var_Impurity(K, K0, K1):
    return (K0 * K1) / K**2

def Entropy(K, K0, K1):
    if (K0 == 0 or K1 == 0):
        return 0
    P_0 = K0 / K
    P_1 = 1 - P_0
    return -P_0 * lg(P_0) - P_1 * lg(P_1)


# In[39]:


if (len(sys.argv) != 3 and len(sys.argv) != 5):
    print('Error: Need either 2 arguments for forests and 5 for decision trees')
    sys.exit(1)    
fileStr              = ''
learner              = None
impurityFunction     = None
impurityFunctionName = ''
pruningAlgorithm     = []

dtNames        = ['decisiontree', 'dt']
rfNames        = ['randomforest', 'rf']
learnerOptions = dtNames + rfNames

entropyNames    = ['e', 'entropy']
viNames         = ['varianceimpurity', 'vi']
impurityOptions = entropyNames + viNames

repNames       = ['reducederrorpruning', 'rep', 'reducederror']
dbpNames       = ['depthbasedpruning', 'dbp', 'depthbased']
naiveNames     = ['naive', 'n']
pruningOptions = repNames + dbpNames + naiveNames

fileStr      = sys.argv[1].lower()

if (sys.argv[2].lower() in learnerOptions):
    learner = sys.argv[2].lower()
else:
    print('Error: Invalid argument for learning algorithm', sys.argv[1])
    sys.exit(1)

if (learner in rfNames):
    if (len(sys.argv) > 3):
        print('Error: Too many arguments for random forest')
        sys.exit(1)
else:
    if (len(sys.argv) < 5):
        print('Error: Insufficient arguments for decision tree(either need more or less)')
        sys.exit(1)

    if (sys.argv[3].lower() in impurityOptions):
        impurityFunction, impurityFunctionName = (Entropy, 'Entropy') if sys.argv[3].lower() in entropyNames else (Var_Impurity, 'Variance Impurity')
    else:
        print('Error: Inavlid argument for impurity function(options: \'entropy\' or \'varianceimpurity\')')
        sys.exit(1)
    pruningAlgorithm = sys.argv[4].replace(' ', '').lower().split(',')
    for option in pruningAlgorithm:
        if not option in pruningOptions:
            print('Error: Option \'', option, 'not valid')
            sys.exit(1)


# In[34]:


trainFilename = "train_" + fileStr + ".csv"
validFilename = "valid_" + fileStr + ".csv"
testFilename = "test_"  + fileStr + ".csv"
train = pd.read_csv("./all_data/" + trainFilename)
valid = pd.read_csv("./all_data/" + validFilename)
test  = pd.read_csv("./all_data/" + testFilename)


# In[11]:


train.columns = valid.columns = test.columns = list(range(0, len(train.columns)))


# In[4]:


#H = Current impurity, X_col = column being tested, S = data set, tA = target attribute, impF = impurity function
#Inf_Gain(H, X_col, S, tA, impF)
#  sum <- 0
#  unique <- unique values in X_col of S
#  for x in unique vals
#    Sn <- Values in S where X_col = x
#    occurences <- number of occurences of each unique value in tA
#    sum <- sum + entropy(occurences)
#  return H - sum
def Inf_Gain(H, X_col, S, tA, impF):
    sum = 0
    K = len(S)
    uniqVals = S[X_col].unique()
    for x in uniqVals:
        Sn  = S[S[X_col] == x]
        occurences = Sn[tA].value_counts()
        K0 = occurences[0] if (0 in occurences.index) else 0
        K1 = occurences[1] if (1 in occurences.index) else 0
        Kv = K0 + K1
        sum = sum + (Kv / K) * impF(Kv, K0, K1)
    return H - sum;

def Get_Best_Attribute(S, tA, cols, impF):
    H = impF(len(S), len(S[S[tA] == 0]), len(S[S[tA] == 1]))
    maxGain = (0, 0)
    for col in cols:
        newGain = Inf_Gain(H, col, S[[tA, col]], tA, impF)
        maxGain = (col, newGain) if (newGain > maxGain[1])  else maxGain
    return maxGain[0]

#S = data, tA = target attribute, cols = columns to test on, 
#impF = impurity function
#Grow_Tree(S,tA, cols, impF)
#  s_uniq <- All unique classes in target attribute
#  if s_uniq has only one value return a leaf with that value
#  else
#    x_j <- Attribute name with highest gain
#    return node(x_j, 
#                Grow_Tree(S with x_j values == 0, tA, cols, impF),
#                Grow_Tree(S with x_j values == 1, tA, cols, impF))
def Grow_Tree(S, tA, cols, impF):
    s_uniq = S[tA].unique();
    if (len(s_uniq) == 1 and s_uniq[0] == 0):
        return (0)
    elif (len(s_uniq) == 1 and s_uniq[0] == 1):
        return (1)
    else:
        x_j = Get_Best_Attribute(S,tA,cols[cols != tA],impF)
        return (x_j,
                Grow_Tree(S[S[x_j] == 0], tA, 
                          cols[cols != x_j], impF), 
                Grow_Tree(S[S[x_j] == 1], tA, 
                          cols[cols != x_j], impF))


# In[5]:


#Predict_Tree()
# While not at leaf:
#  If value has no nodes on left or right, break
#  Find attribute at current node x_j
#  Go left if value at attribute x_j == 0, otherwise go right
# pred = leaf.val
def pred_tree(tree, data):
    trav_tree = copy.copy(tree)
    atLeaf = False
    while True:
        if trav_tree == 0 or trav_tree == 1:
            break;
        if data[trav_tree[0]] == 0:
            trav_tree = trav_tree[1]
        else:
            trav_tree = trav_tree[2]
    return trav_tree
        
def compare_result(tree, data, tA):
    pred = pred_tree(tree, data)
    return pred == data[tA]


# In[12]:


#Need to change nodes from splits to leafs
#Evaluate starting from nodes holding leaf nodes if they should be replaced
#Continue upwards, with next level of nodes being nodes holding the nodes that are holding leafs

#V = data set, tA = target attribute
#Reduced_Error_Prune_Helper(V,tA, tree)
#Start from bottom nodes and continuously replace those nodes with more general nodes.
#currTree keeps track of branch needed to be replaced.
#Position can help reach a node in the tree
#treeAndAcc is a tuple with the tree on the lefthand side, and accuracy and on the right.
def Reduced_Error_Prune(V, tA, tree):
    treec = copy.copy(tree)
    treeAndAcc = (tree, Get_Tree_Accuracy(V, tA, tree))
    return Reduced_Error_Prune_Helper(V, tA, treeAndAcc, ())[0]

#2^(tree height) = total node estimate in tree.
#O(Size of the data * Numbers of nodes in the tree - Number of leafs)
def Reduced_Error_Prune_Helper(V, tA, treeAndAcc, position):
    currTree = Traverse_Tree(treeAndAcc[0], position)
    
    if (currTree == 1 or currTree == 0):
        return treeAndAcc
    if (type(currTree[1]) == tuple):
        treeAndAcc = Reduced_Error_Prune_Helper(V, tA, 
                                                  treeAndAcc, 
                                                  position + (0,))
    if (type(currTree[2]) == tuple):
        treeAndAcc = Reduced_Error_Prune_Helper(V, tA, 
                                                  treeAndAcc, 
                                                  position + (1,))
    
    mcl         = Most_Common_Leaf(currTree)
    newTree     = Set_Tree(treeAndAcc[0], mcl, copy.copy(position))
    newAccuracy = Get_Tree_Accuracy(V, tA, newTree)
    
    return (newTree, newAccuracy) if newAccuracy >= treeAndAcc[1] else treeAndAcc

def Depth_Based_Prune(V, tA, tree):
    dMax        = [5,10,15,20,50,100]
    bestTree    = copy.copy(tree)
    maxAccuracy = Get_Tree_Accuracy(V, tA, tree)
    
    for depth in dMax:
        newTree = Prune_On_Depth(tree, depth, 0)
        newAccuracy  = Get_Tree_Accuracy(V, tA, newTree)
        (bestTree, maxAccuracy) = (newTree, newAccuracy) if newAccuracy > maxAccuracy else (bestTree, maxAccuracy)
    return bestTree

def Prune_On_Depth(tree, dMax, depth):
    depth += 1
    if (tree == 1 or tree == 0):
        return tree
    if (depth < dMax):
        if (type(tree[1]) == tuple):
            tree = (tree[0], Prune_On_Depth(tree[1], dMax, depth), tree[2])
        if (type(tree[2]) == tuple):
            tree = (tree[0], tree[1], Prune_On_Depth(tree[2], dMax, depth))
        return tree
    else:
        return Most_Common_Leaf(tree)
    
#Traverse and return a node in a tree based on a position array
def Traverse_Tree(tree, position):
    nTree = copy.copy(tree)
    for i in position:
        if (i == 0):
            nTree = nTree[1]
        else:
            nTree = nTree[2]
    return nTree;

#Replace a node in a tree by its position and return the full tree.
def Set_Tree(tree, newSubTree, position):
    if (len(position) == 0):
        return newSubTree
    nextPos = position[0]
    position = position[1:]
    if (nextPos == 0):
        return (tree[0], Set_Tree(tree[1],newSubTree,position), tree[2])
    if (nextPos == 1):
        return (tree[0], tree[1], Set_Tree(tree[2],newSubTree,position))

#Returns most common leaf from a tree/subtree.
def Most_Common_Leaf(tree):
    leafs = Get_All_Leafs(tree)
    return max(leafs, key=leafs.count)

#Returns a list of all the leafs in a tree
def Get_All_Leafs(tree):
    if (type(tree) == tuple):
        return Get_All_Leafs(tree[1]) + Get_All_Leafs(tree[2])
    else:
        if (tree == 0):
            return [0]
        else:
            return [1]
    
#Evaluates accuracy of a tree on target attribute tA based on data D
def Get_Tree_Accuracy(D, tA, tree):
    results = D.apply(lambda instance: compare_result(tree, instance, tA), axis=1)
    return len(results[results == True])/len(results)


# In[15]:


def Analyze_Prune_Trees(V, T, tA, tree):
    for prune in pruningAlgorithm:
        if prune in naiveNames:
            print("Accuracy with naive trees trained on " + impurityFunctionName +
                  " in " + testFilename + ":"
                  ,Get_Tree_Accuracy(T, tA, tree))
        elif prune in dbpNames:
            dbpTree = Depth_Based_Prune(V, tA, tree)
            print("Accuracy with trees using depth-based pruning trained on " + impurityFunctionName +
                  " in " + testFilename + ":"
                  ,Get_Tree_Accuracy(T, tA, dbpTree))
        elif prune in repNames:
            repTree = Reduced_Error_Prune(V, tA, tree)
            print("Accuracy with trees using reduced-error pruning trained on " + impurityFunctionName +
                  "in " + testFilename + ":"
                  ,Get_Tree_Accuracy(T, tA, repTree))            


# In[42]:


if (learner in dtNames):
    print("Growing tree on " + trainFilename)
    startTime = datetime.now()
    tree = Grow_Tree(train, 1, train.columns, impurityFunction)
    print("Time taken to grow tree: ", datetime.now() - startTime)
    Analyze_Prune_Trees(valid, test, 1, tree)
else:
    from sklearn.model_selection import train_test_split

    X = train.drop(columns=[1])
    y = train[1]

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X, y)
    predict = model.predict(test.drop(columns=[1]))

    from sklearn.metrics import accuracy_score

    print("Accuracy with random forests in " + testFilename + ":", accuracy_score(test[1], predict))

