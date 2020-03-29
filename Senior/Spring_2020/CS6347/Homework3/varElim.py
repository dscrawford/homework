import itertools
from collections import Counter
import numpy as np
import regex as re
from os import path
import random
import sys
import argparse

# parser = argparse.ArgumentParser(description='Variable Elimination with sampling')
# parser.add_argument('filename', metavar='filename', type=str, help='A filename with its path(no extensions)')
# parser.add_argument('w', metavar='w', type=int, help='W-cutset value')
# parser.add_argument('N', metavar='N', type=int, help='Number of samples to take')
# parser.add_argument('--random_seed', metavar='random', type=int, help='Random seed, like it says')
# parser.add_argument('--adaptive', metavar='adaptive', type=bool,
#                     help='Whether to use adaptive distribution or not')
#
# args = parser.parse_args()
# fileName = args.filename
# w = args.w
# N = args.N
# random_seed = args.random_seed
# adaptive = args.adaptive
#
# random.seed(random_seed)
np.seterr(all='ignore')
log = np.log10
base = 10


def logsumexp(a):
    mx = log(max(a))
    return log(np.sum([threshold(base ** (log(i) - mx)) for i in a])) + mx


def threshold(x):
    if x < 1e-10:
        return 0.0
    elif x == float('inf'):
        return sys.float_info.max
    return x


class Factor(object):
    card = []
    cliqueScope = []
    functionTable = []
    stride = []

    def __init__(self, cliqueScope=None, functionTable=None, card=None):
        self.cliqueScope = np.array(cliqueScope)
        self.card = np.array(card)
        self.stride = self.getStride(cliqueScope)
        self.functionTable = functionTable

    def getIndex(self, assignment: np.array):
        if len(assignment) == 0:
            return 0
        return sum(assignment * self.stride)

    @staticmethod
    def getAssignment(index: int, stride: int, card: int):
        return index // stride % card

    def getAssignments(self, index: int):
        if len(self.stride) == 0:
            return 0
        return [self.getAssignment(index, self.stride[i], self.card[self.cliqueScope[i]]) for i in
                range(len(self.stride))]

    def getStride(self, cliqueScope):
        prod = 1
        n = len(cliqueScope)
        a = [0 for i in range(n)]
        for i in range(n):
            a[i] = prod
            prod = prod * self.card[i]
        return list(a)

    def factorProduct(self, F1, F2):
        clique = np.array(list(set().union(F1.cliqueScope, F2.cliqueScope)))
        x1i = np.ndarray.flatten(np.array([np.argwhere(clique == i) for i in F1.cliqueScope])).astype(np.int64)
        x2i = np.ndarray.flatten(np.array([np.argwhere(clique == i) for i in F2.cliqueScope])).astype(np.int64)
        vn = np.product(self.card[clique])
        F3 = Factor(clique, np.full(vn, 0.0), self.card)
        for i in range(vn):
            assign = np.array(F3.getAssignments(i))
            x1 = F1.functionTable[F1.getIndex(assign[x1i])]
            x2 = F2.functionTable[F2.getIndex(assign[x2i])]
            F3.functionTable[i] = threshold(x1 * x2)
        return F3

    def sumVariable(self, v):
        X = self.cliqueScope
        F = self.functionTable
        S = self.stride
        vi = int(np.argwhere(X == v))
        n = np.product(self.card[X])
        newCs = [x for x in X if x != v]
        newPhi = Factor(newCs, [[] for _ in range(n // self.card[v])], self.card)
        for i in range(n):
            assignment = self.getAssignments(i)
            newPhi.functionTable[newPhi.getIndex(np.delete(assignment, vi))] += [F[i]]
        newPhi.functionTable = [threshold(base ** logsumexp(i)) for i in newPhi.functionTable]
        return newPhi

    def instantiateEvidence(self, evidence):
        newFactor = Factor(self.cliqueScope, self.functionTable, self.card)
        for var, val in evidence:
            varI = np.argwhere(newFactor.cliqueScope == var)
            if len(varI) != 1:
                continue
            varI = int(varI)
            newI = []
            for j in range(len(newFactor.functionTable)):
                assignment = newFactor.getAssignments(j)
                if assignment[varI] == val:
                    newI.append(j)
            cliqueScope = np.delete(newFactor.cliqueScope, varI)
            functionTable = newFactor.functionTable[newI]
            newFactor = Factor(cliqueScope, functionTable, self.card)
        return newFactor

    def __mul__(self, other):
        return self.factorProduct(self, other)


class GraphicalModel:
    networkType = ""
    varN = 0
    cliques = 0
    evidence = np.array([])
    minDegreeOrder = []
    factors = []
    card = []
    Z = 1

    def __init__(self, uaiFile: str):
        self.parseUAI(uaiFile + ".uai")
        if path.exists(uaiFile + ".uai.evid"):
            self.parseUAIEvidence(uaiFile + ".uai.evid")
            self.factors = self.instantiateEvidence(self.evidence)
        self.minDegreeOrder = self.getOrder()

    def getOrder(self, factors=None):
        if factors is None:
            factors = self.factors
        cliqueSets = [set(f.cliqueScope) for f in factors]
        clique = set()
        for cs in cliqueSets:
            clique = clique | cs
        varD = {v: set() for v in clique}
        for v in clique:
            for cs in cliqueSets:
                if v in cs:
                    varD[v] = varD[v] | cs
        order = []
        # Iterate through all edges now, select min degree variable and all add those edges to each other variable that
        # the edge was connected to.
        for i in range(len(clique)):
            minVar = list(varD.keys())[0]
            minDegree = len(varD[minVar])
            for var in clique:
                if len(varD[var]) < minDegree:
                    minVar = var
                    minDegree = len(varD[minVar])
            order.append(minVar)
            for var in clique:
                if minVar in varD[var]:
                    varD[var] = varD[var] | varD[minVar]
                    varD[var] = varD[var] - {minVar}
            del (varD[minVar])
            clique.remove(minVar)
        return order

    def sumOut(self, factors=None, minDegreeOrder=None):
        if factors is None:
            factors = self.factors
        if minDegreeOrder is None:
            minDegreeOrder = self.minDegreeOrder
        functions = [f for f in factors]
        for o in minDegreeOrder:
            phi = [f for f in functions if o in f.cliqueScope]
            functions = [f for f in functions if o not in f.cliqueScope]
            newPhi = phi.pop()
            for p in phi:
                newPhi = newPhi * p
            functions.append(newPhi.sumVariable(o))
        return np.sum(log([f.functionTable[0] for f in functions]))

    def instantiateEvidence(self, evidence):
        return [self.factors[i].instantiateEvidence(evidence) for i in range(self.cliques)]

    def sampleSumOut(self, w, N):
        X = self.wCutset(w)
        S = []
        Q = []
        Qa = []
        VE = []
        minDegreeOrder = [o for o in self.minDegreeOrder if o not in X]
        for i in range(N):
            sampleEvidence = self.generateSampleUniform(X)
            S.append(sampleEvidence)
            z = self.sumOut(self.instantiateEvidence(sampleEvidence), minDegreeOrder)
            VE.append(z)
            q = sum([log(1 / self.card[var]) for var in X])
            Q.append(q)
            Qa.append(q)
            if 0 == (i + 1) % 100 and i + 1 > 1:
                Qa = self.adaptiveQ(Qa, VE, S)
        return (sum([VE[i] - Q[i] for i in range(N)]) / N), (sum([VE[i] - Qa[i] for i in range(N)]) / N)

    def generateSampleUniform(self, X: set):
        return [(var, int(random.uniform(0, self.card[var]))) for var in X]

    @staticmethod
    def adaptiveQ(Q, VE, S):
        n = len(Q)
        Qn = [0 for _ in range(n)]
        W = [VE[i] - Q[i] for i in range(n)]
        Wtot = base ** logsumexp(W)
        for i, s1 in enumerate(S):
            Qn[i] = sum([(1 if s1 == S[j] else 0) * W[j] / Wtot for j in range(n)])
        return Qn

    # wCutset(C, m, w): where C is the cliques and m is the min-degree ordering
    # let t be an empty tree
    # t <- schematic bucket elimination induced tree
    # X = empty set
    # while largest set in tree is larger than w+1
    #  v <- variable that appears most often in tree
    #  t <- t without v at each cluster
    #  X = X U v
    # return X
    def wCutset(self, w):
        cliqueScopes = [f.cliqueScope for f in self.factors]
        tree = []
        for o in self.minDegreeOrder:
            treeNode = [cs for cs in cliqueScopes if o in cs]
            nodes = set()
            for cs in treeNode:
                for var in cs:
                    nodes = nodes.union({var})
            tree.append(list(nodes))
            cliqueScopes = [cs for cs in cliqueScopes if o not in cs]
            cliqueScopes.append(list(nodes - {o}))
        X = set()
        while np.max([len(a) for a in tree]) > w + 1:
            l = list(itertools.chain(*tree))
            data = Counter(l)
            v = max(l, key=data.get)
            tree = [[c for c in cs if c != v] for cs in tree]
            X = X.union({v})
        return X

    def parseUAIEvidence(self, evidenceFile: str):
        s = [t for t in open(evidenceFile, "r").read().split(' ') if t]
        observedVariables = int(s.pop(0))
        self.evidence = [(int(s[2 * i]), int(s[2 * i + 1])) for i in range(0, observedVariables)]

    def parseUAI(self, uaiFile: str):
        s = re.sub('^c.*\n?', '', open(uaiFile, "r").read(), flags=re.MULTILINE)
        s = [l for l in s.split('\n') if l]
        s = [l.replace('\t', ' ').replace('\r', ' ').replace('\f', ' ').replace('\v', ' ') for l in s]
        # Below parses the UAI file
        card = []
        cliqueScopes = []
        functionTables = []

        while len(s) != 0:
            data = s.pop(0)
            if data.upper() in {'MARKOV', 'BAYES'}:
                self.networkType = data
                self.varN = int(s.pop(0))
                card = np.array([int(d) for d in s.pop(0).split(' ') if d])
                self.cliques = int(s.pop(0))
                cliqueScopes = []
                while True:
                    cs = s.pop(0)
                    cliqueScopes += [[int(d) for d in cs.split(' ') if d][1:]]
                    if len([t for t in s[0] if t]) == 1:
                        break
            if data.isdigit():
                for i in range(self.cliques):
                    entriesN = int(data)
                    entries = []
                    entriesAdded = 0
                    while entriesAdded != entriesN:
                        newEntries = [float(d) for d in s.pop(0).split(' ') if d]
                        entriesAdded += len(newEntries)
                        entries += newEntries
                    functionTables += [np.array(entries).astype(np.float64)]
                    data = None if i == self.cliques - 1 else s.pop(0)
        self.factors = [Factor(cliqueScopes[i], functionTables[i], card) for i in range(self.cliques)]
        self.card = card
