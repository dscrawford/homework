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
        return [index // stride[i] % card[i] for i in range(len(stride) - 1, 0, -1)]

    def factorProduct(self, f1, f2):
        j = 0
        k = 0
        vars = list(set().union(self.cliqueScopes[f1], self.cliqueScopes[f2]))
        vn = np.product(self.card[vars])
        n = np.max(self.card[vars])
        psi = np.full(vn, 0.0)
        # for i in range(0, vn):
        #     psi[i] = self.functionTables[f1][j] * self.functionTables[f2][k]
        #     for l in range(0, n):
        #         assignment[l] = assignment[l] + 1
        #         if assignment[l] == self.card[l]:
        #             assignment[l] = 0
        #             j = j - (self.card[l] - 1) * (0 if l >= len(self.stride[f1]) else self.stride[f1][l])
        #             k = k - (self.card[l] - 1) * (0 if l >= len(self.stride[f2]) else self.stride[f2][l])
        #         else:
        #             j = j + (0 if l >= len(self.stride[f1]) else self.stride[f1][l])
        #             k = k + (0 if l >= len(self.stride[f2]) else self.stride[f2][l])
        #             break

        return psi

    def instantiateEvidence(self):
        print('err')

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
            self.stride = np.array([[np.product(self.card[cs[range(0, i)]]) for i, _ in enumerate(cs)]
                                    for cs in self.cliqueScopes])


network = GraphicalModel("network")
print(network.factorProduct(0, 1))
