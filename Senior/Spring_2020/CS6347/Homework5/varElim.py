import itertools
from collections import Counter, defaultdict
import numpy as np
import regex as re
from os import path
import random
import sys
from copy import deepcopy
from logdouble import Log_Double

np.seterr(all='ignore')


# def logsumexp(a):
#     mx = log(max(a))
#     return log(np.sum([threshold(base ** (log(i) - mx)) for i in a])) + mx
#
#
# def threshold(x):
#     if x < 1e-10:
#         return 0.0
#     elif x == float('inf'):
#         return sys.float_info.max
#     return x


class Factor:
    card = {}
    cliqueScope = []
    functionTable = []
    stride = []

    def __init__(self, cliqueScope=None, functionTable=None, card=None):
        self.cliqueScope = np.array(cliqueScope)
        self.card = card
        self.stride = self.getStride(cliqueScope)
        if functionTable is None:
            self.functionTable = np.full(int(np.product([card[var] for var in cliqueScope])), 0)
        else:
            self.functionTable = functionTable

    def getIndex(self, assignment: np.array):
        if len(assignment) == 0:
            return 0
        return sum(assignment * list(reversed(self.stride)))

    @staticmethod
    def getAssignment(index: int, stride: int, card: int):
        return index // stride % card

    def getAssignments(self, index: int):
        if len(self.stride) == 0:
            return 0
        return [self.getAssignment(index, self.stride[-(i+1)], self.card[self.cliqueScope[-(i+1)]]) for i in
                range(len(self.stride))]

    def getAssignmentsAsEvidence(self, index: int):
        assign = self.getAssignments(index)
        return {self.cliqueScope[i]: assign[i] for i in range(len(self.cliqueScope))}

    def getStride(self, cliqueScope):
        prod = 1
        a = []
        for i in range(len(cliqueScope)):
            a.append(prod)
            prod = prod * self.card[cliqueScope[i]]
        return a

    def getSize(self):
        if len(self.cliqueScope) == 0:
            return 1
        return int(np.product([self.card[var] for var in self.cliqueScope]))

    def factorProduct(self, F1, F2):
        clique = np.array(list(set().union(F1.cliqueScope, F2.cliqueScope)))
        f1_clique_set = set(F1.cliqueScope)
        f2_clique_set = set(F2.cliqueScope)
        card = F1.card.copy()
        card.update(F2.card)
        x1i = [i for i in range(len(clique)) if clique[i] in f1_clique_set]
        x2i = [i for i in range(len(clique)) if clique[i] in f2_clique_set]
        vn = np.product([card[var] for var in clique])
        F3 = Factor(clique, np.full(vn, Log_Double()), card)
        for i in range(vn):
            assign = np.array(F3.getAssignments(i))
            x1 = F1.functionTable[F1.getIndex(assign[x1i])]
            x2 = F2.functionTable[F2.getIndex(assign[x2i])]
            F3.functionTable[i] = (x1 * x2)
        return F3

    def sumVariable(self, v):
        vi = int(np.argwhere(self.cliqueScope == v))
        n = len(self.functionTable)
        newCs = [x for x in self.cliqueScope if x != v]
        newcard = self.card.copy()
        del(newcard[v])
        newPhi = Factor(newCs, [Log_Double() for _ in range(n // self.card[v])], newcard)
        for i in range(n):
            assign = self.getAssignments(i)
            index = newPhi.getIndex(np.delete(assign, vi))
            newPhi.functionTable[index] += self.functionTable[i]
        return newPhi

    def instantiateEvidence(self, evidence):
        evid_i = []
        non_evid_i = []
        for j in range(len(self.cliqueScope)):
            if self.cliqueScope[j] in evidence:
                evid_i.append(j)
            else:
                non_evid_i.append(j)
        new_clique_scope = [cs for cs in self.cliqueScope if cs not in evidence]
        new_n = int(np.product([self.card[cs] for cs in new_clique_scope]))
        new_card = self.card.copy()
        for vari in evid_i:
            del(new_card[self.cliqueScope[vari]])
        newFactor = Factor(new_clique_scope, np.full(new_n, Log_Double()), new_card)
        assign = np.full(len(self.cliqueScope), 0)
        assign[evid_i] = [evidence[self.cliqueScope[i]] for i in evid_i]
        for i in range(new_n):
            assign[non_evid_i] = newFactor.getAssignments(i)
            newFactor.functionTable[i] = self.functionTable[self.getIndex(assign)]
        return newFactor

    def __mul__(self, other):
        return self.factorProduct(self, other)


class Network:
    networkType = ""
    varN = 0
    cliques = 0
    evidence = {}
    factors = []
    card = {}

    def __init__(self, file_name, ignore_factors=False):
        if file_name is not None:
            self.parseUAI(file_name + ".uai", ignore_factors=ignore_factors)
            if path.exists(file_name + ".uai.evid"):
                self.parseUAIEvidence(file_name + ".uai.evid")

    def create_network(self, networkType, varN, cliques, factors, card):
        self.networkType = networkType
        self.varN = varN
        self.cliques = cliques
        self.factors = factors
        self.card = card
        return self

    def parseUAIEvidence(self, evidenceFile: str):
        s = [t for t in open(evidenceFile, "r").read().split(' ') if t]
        observedVariables = int(s.pop(0))
        self.evidence = {int(s[2 * i]): int(s[2 * i + 1]) for i in range(observedVariables)}

    def parseUAI(self, uaiFile: str, ignore_factors=False):
        s = re.sub('^c.*\n?', '', open(uaiFile, "r").read(), flags=re.MULTILINE)
        s = [l for l in s.split('\n') if l]
        s = [l.replace('\t', ' ').replace('\r', ' ').replace('\f', ' ').replace('\v', ' ') for l in s]
        # Below parses the UAI file
        card = {}
        cliqueScopes = []
        functionTables = []

        while len(s) != 0:
            data = s.pop(0)
            if data.upper() in {'MARKOV', 'BAYES'}:
                self.networkType = data
                self.varN = int(s.pop(0))
                card = [int(d) for d in s.pop(0).split(' ') if d]
                card = {var: card[var] for var in range(len(card))}
                self.cliques = int(s.pop(0))
                cliqueScopes = []
                for _ in range(self.cliques):
                    cs = s.pop(0)
                    cliqueScopes += [[int(d) for d in cs.split(' ') if d][1:]]

            if data.isdigit() and not ignore_factors:
                for i in range(self.cliques):
                    entriesN = int(data)
                    entries = []
                    entriesAdded = 0
                    while entriesAdded != entriesN:
                        newEntries = [Log_Double(float(d)) for d in s.pop(0).split(' ') if d]
                        entriesAdded += len(newEntries)
                        entries += newEntries
                    functionTables += [np.array(entries)]
                    data = None if i == self.cliques - 1 else s.pop(0)
        if not ignore_factors:
            self.factors = [Factor(cliqueScopes[i], functionTables[i], {var: card[var] for var in cliqueScopes[i]})
                            for i in range(self.cliques)]
        else:
            self.factors = [Factor(cliqueScopes[i], None, {var: card[var] for var in cliqueScopes[i]}) for i in range(self.cliques)]
        self.card = card
        self.cliqueScopes = cliqueScopes


class GraphicalModel:
    networkType = ""
    varN = 0
    cliques = 0
    evidence = {}
    minDegreeOrder = []
    factors = []
    card = {}

    def __init__(self, network):
        self.networkType = network.networkType
        self.varN = network.varN
        self.cliques = network.cliques
        self.evidence = network.evidence
        self.card = network.card
        self.factors = network.factors
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

    def getLikelihood(self, factors=None, order=None, evidence={}):
        if factors is None:
            factors = self.factors
        if order is None:
            order = self.minDegreeOrder
        if len(evidence) > 0:
            factors = self.instantiateEvidence(evidence)
        functions = [Factor(f.cliqueScope, f.functionTable, f.card) for f in factors]
        order = [o for o in order if o not in evidence and o not in self.evidence]
        for o in order:
            phi = [f for f in functions if o in f.cliqueScope]
            functions = [f for f in functions if o not in f.cliqueScope]
            if len(phi) > 0:
                newPhi = phi.pop()
                for p in phi:
                    newPhi = newPhi * p
                functions.append(newPhi.sumVariable(o))
        return np.product([f.functionTable[0] for f in functions])

    def instantiateEvidence(self, evidence):
        return [self.factors[i].instantiateEvidence(evidence) for i in range(self.cliques)]

    def sampleGetLikelihood(self, w, N):
        X = self.wCutset(w)
        S = []
        Q = []
        Qa = []
        VE = []
        minDegreeOrder = [o for o in self.minDegreeOrder if o not in X]
        for i in range(N):
            sampleEvidence = self.generateSampleUniform(X)
            S.append(sampleEvidence)
            z = self.getLikelihood(evidence=sampleEvidence, order=minDegreeOrder)
            VE.append(z)
            q = sum([Log_Double(1 / self.card[var]) for var in X])
            Q.append(q)
            Qa.append(q)
            if 0 == (i + 1) % 100 and i + 1 > 1:
                Qa = self.adaptiveQ(Qa, VE, S)
        return (sum([VE[i] - Q[i] for i in range(N)]) / N), (sum([VE[i] - Qa[i] for i in range(N)]) / N)

    def generateSampleUniform(self, X: set):
        return {var: int(random.uniform(0, self.card[var])) for var in X}

    @staticmethod
    def adaptiveQ(Q, VE, S):
        n = len(Q)
        Qn = [Log_Double() for _ in range(n)]
        W = [VE[i] - Q[i] for i in range(n)]
        Wtot = sum(W)
        for i, s1 in enumerate(S):
            Qn[i] = sum([(1 if s1 == S[j] else 0) * W[j] / Wtot for j in range(n)])
        return Qn
    #
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