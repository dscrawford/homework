import itertools
from collections import Counter
import numpy as np
import regex as re
from os import path
from sys import argv

# if len(argv) != 2:
#     print('Error: Insufficient amount of arguments')
#     exit(1)
# fileName = argv[1]


class Factor(object):
    card = []
    cliqueScope = []
    functionTable = []
    stride = []

    def __init__(self, cliqueScope, card):
        self.cliqueScope = np.array(cliqueScope)
        self.card = np.array(card)
        self.stride = self.getStride(cliqueScope)

    def __init__(self, cliqueScope, functionTable, card):
        self.cliqueScope = np.array(cliqueScope)
        self.functionTable = functionTable
        self.card = np.array(card)
        self.stride = self.getStride(cliqueScope)

    def getIndex(self, assignment: np.array):
        if len(assignment) == 0:
            return 0
        return sum(assignment * self.stride)

    def getAssignment(self, index: int, stride: int, card: int):
        return index // stride % card

    def getAssignments(self, index: int):
        if (len(self.stride) == 0):
            return 0
        return [self.getAssignment(index, self.stride[i], self.card[self.cliqueScope[i]]) for i in range(len(self.stride))]

    def getStride(self, cliqueScope):
        prod = 1
        a = []
        for i in reversed(cliqueScope):
            a.append(prod)
            prod = prod * self.card[i]
        return list(reversed(a))

    def factorProduct(self, F1, F2):
        clique = np.array(list(set().union(F1.cliqueScope, F2.cliqueScope)))
        stride = self.getStride(np.array(clique))
        x1i = np.ndarray.flatten(np.array([np.argwhere(clique == i) for i in F1.cliqueScope])).astype(np.int64)
        x2i = np.ndarray.flatten(np.array([np.argwhere(clique == i) for i in F2.cliqueScope])).astype(np.int64)
        vn = np.product(self.card[clique])
        F3 = Factor(clique, np.full(vn, 0.0), self.card)
        for i in range(vn):
            assign = np.array(F3.getAssignments(i))
            F3.functionTable[i] = F1.functionTable[F1.getIndex(assign[x1i])] * F2.functionTable[F2.getIndex(assign[x2i])]
        return F3

    def sumVariable(self, v):
        X = self.cliqueScope
        F = self.functionTable
        S = self.stride
        vi = int(np.argwhere(X == v))
        n = np.product(self.card[X])
        newCs = [x for x in X if x != v]
        newPhi = Factor(newCs, np.full(n // self.card[v], 0.0), self.card)
        for i in range(np.product(self.card[X])):
            assignment = self.getAssignments(i)
            newPhi.functionTable[newPhi.getIndex(np.delete(assignment, vi))] += F[i]
        return newPhi

    def instantiateEvidence(self, evidence):
        for var, val in evidence:
            varI = np.argwhere(self.cliqueScope == var)
            if len(varI) != 1:
                continue
            varI = int(varI)
            newI = []
            for j in range(len(self.functionTable)):
                assignment = self.getAssignments(j)
                if assignment[varI] == val:
                    newI.append(j)
            self.functionTable = self.functionTable[newI]
            self.cliqueScope = np.delete(self.cliqueScope, varI)
            self.stride = self.getStride(self.cliqueScope)

    def __mul__(self, other):
        return self.factorProduct(self, other)


class GraphicalModel:
    networkType = ""
    varN = 0
    cliques = 0
    evidence = np.array([])
    order = []
    factors = []

    def __init__(self, uaiFile: str):
        self.parseUAI(uaiFile + ".uai")
        if (path.exists(uaiFile + ".uai.evid")):
            self.parseUAIEvidence(uaiFile + ".uai.evid")
            self.instantiateEvidence()
        self.order = self.getOrder()
        print(self.wCutset(1))

    def getOrder(self):
        cliqueSets = [set(f.cliqueScope) for f in self.factors]
        vars = set()
        for cs in cliqueSets:
            vars = vars | cs
        varD = {v: set() for v in vars}
        for v in vars:
            for cs in cliqueSets:
                if v in cs:
                    varD[v] = varD[v] | cs
        order = []
        # Iterate through all edges now, select min degree variable and all add those edges to each other variable that
        # the edge was connected to.
        for i in range(len(vars)):
            minVar = list(varD.keys())[0]
            minDegree = len(varD[minVar])
            for var in vars:
                if len(varD[var]) < minDegree:
                    minVar = var
                    minDegree = len(varD[minVar])
            order.append(minVar)
            for var in vars:
                if minVar in varD[var]:
                    varD[var] = varD[var] | varD[minVar]
                    varD[var] = varD[var] - {minVar}
            del (varD[minVar])
            vars.remove(minVar)
        return order

    def sumOut(self):
        functions = [f for f in self.factors]
        for o in self.order:
            phi = [f for f in functions if o in f.cliqueScope]
            functions = [f for f in functions if o not in f.cliqueScope]
            newPhi = None
            if len(phi) == 0:
                continue
            elif len(phi) > 1:
                newPhi = phi[0]
                phi.pop(0)
                for p in phi:
                    newPhi = newPhi * p
            else:
                newPhi = phi[0]
            functions.append(newPhi.sumVariable(o))
        return np.sum(np.log10([f.functionTable[0] for f in functions]))

    def instantiateEvidence(self):
        for f in self.factors:
            f.instantiateEvidence(self.evidence)

    def sampleSumOut(self,w,N):
        Z = 0
        X = self.wCutset(w)

    def wCutset(self, w):
        cliqueScopes = [f.cliqueScope for f in self.factors]
        tree = []
        for o in self.order:
            tree.append(list(np.unique([cs for cs in cliqueScopes if o in cs])))
            cliqueScopes = [cs for cs in cliqueScopes if o not in cs]
        X = set()
        while np.max([len(a) for a in tree]) > w + 1:
            l = list(itertools.chain(*tree))
            data = Counter(l)
            max(l, key=data.get)
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
                    functionTables += [np.array(entries)]
                    data = None if i == self.cliques - 1 else s.pop(0)
        self.factors = [Factor(cliqueScopes[i], functionTables[i], card) for i in range(self.cliques)]


network = GraphicalModel("test")
#print(network.sumOut())
