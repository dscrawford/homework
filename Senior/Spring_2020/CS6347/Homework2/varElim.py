import numpy as np
import pandas as pd
import regex as re
from itertools import combinations

class GraphicalModel:
    networkType = ""
    varN = 0
    card = np.array([])
    cliques = 0
    cliqueScopes = np.array([])
    functionTables = []
    stride = []
    evidence = np.array([])

    def __init__(self, uaiFile: str):
        self.parseUAI(uaiFile + ".uai")
        self.parseUAIEvidence(uaiFile + ".uai.evid")

    def getIndex(self, assignment: np.array, stride: np.array):
        return sum(assignment * stride)

    def getAssignment(self, index: int, stride: int, card: int):
        return index // stride % card

    def getAssignments(self, index: int, stride: np.array, card: np.array):
        return [self.getAssignment(index, stride[i], card[i]) for i in range(len(stride))]

    def getStride(self, cliqueScope):
        prod = 1
        a    = []
        for i in reversed(cliqueScope):
            a.append(prod)
            prod = prod * self.card[i]
        return list(reversed(a))

    def factorProduct(self, x1, x2):
        j    = 0
        k    = 0
        X1   = self.cliqueScopes[x1]
        X2   = self.cliqueScopes[x2]
        clique = np.array(list(set().union(X1, X2)))
        stride = self.getStride(np.array(clique))
        x1i  = np.ndarray.flatten(np.array([np.argwhere(clique == i) for i in X1]))
        x2i  = np.ndarray.flatten(np.array([np.argwhere(clique == i) for i in X2]))
        vn   = np.product(self.card[clique])
        psi  = np.full(vn,0.0)
        for i in range(np.product(self.card[clique])):
            assign = np.array(self.getAssignments(i, stride, self.card[clique]))
            psi[i] = self.functionTables[x1][self.getIndex(assign[x1i], self.stride[x1])] \
                     * self.functionTables[x2][self.getIndex(assign[x2i], self.stride[x2])]
        return (psi, clique, stride)

    # Work in progress..
    def instantiateEvidence(self):
        for var, val in self.evidence:
            for i, cs in enumerate(self.cliqueScopes):
                vari = int(np.argwhere(cs == var))
                newI = []
                for j in range(len(self.functionTables[0])):
                    assignment = self.getAssignment(j, self.stride[i], self.card[i])
                    if (assignment[vari] == val):
                        newI.append(j)
                self.functionTables[i] = self.functionTables[i][newI]
                self.cliqueScopes[i] = np.delete(self.cliqueScopes[i], np.where(self.cliqueScopes[i] == var), axis=0)

    def parseUAIEvidence(self, evidenceFile: str):
        s = [t for t in open(evidenceFile, "r").read().split(' ') if t]
        observedVariables = int(s.pop(0))
        self.evidence = [(int(s[2 * i]), int(s[2 * i + 1])) for i in range(0, observedVariables)]

    def parseUAI(self, uaiFile: str):
        s = re.sub('^c.*\n?', '', open(uaiFile, "r").read(), flags=re.MULTILINE)
        s = [l for l in s.split('\n') if l]
        # Below parses the UAI file
        while (len(s) != 0):
            data = s.pop(0)
            if (data.upper() in ['MARKOV', 'BAYES']):
                self.networkType = data
                self.varN = int(s.pop(0))
                self.card = np.array([int(d) for d in s.pop(0).split(' ') if d])
                self.cliques = int(s.pop(0))
                cliqueScopes = []
                while (True):
                    cs = s.pop(0)
                    cliqueScopes += [[int(d) for d in cs.split(' ') if d][1:]]
                    if (len([t for t in s[0] if t]) == 1):
                        break
                self.cliqueScopes = np.array(cliqueScopes)
            if (data.isdigit()):
                entriesN = int(data)
                entries = []
                entriesAdded = 0
                while (entriesAdded != entriesN):
                    newEntries = [float(d) for d in s.pop(0).split(' ') if d]
                    entriesAdded += len(newEntries)
                    entries += newEntries
                self.functionTables += [np.array(entries)]
            self.stride = np.array([self.getStride(cs) for cs in self.cliqueScopes])


network = GraphicalModel("network")
print(network.factorProduct(0, 1))
