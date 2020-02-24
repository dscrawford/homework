import numpy as np
import regex as re
from os import path


class GraphicalModel:
    networkType = ""
    varN = 0
    card = np.array([])
    cliques = 0
    cliqueScopes = np.array([])
    functionTables = []
    stride = []
    evidence = np.array([])
    order = []

    def __init__(self, uaiFile: str):
        self.parseUAI(uaiFile + ".uai")
        if (path.exists(uaiFile + ".uai.evid")):
            self.parseUAIEvidence(uaiFile + ".uai.evid")
            self.instantiateEvidence()
        self.order = self.getOrder()

    def getIndex(self, assignment: np.array, stride: np.array):
        if len(assignment) == 0:
            return 0
        return sum(assignment * stride)

    def getAssignment(self, index: int, stride: int, card: int):
        return index // stride % card

    def getAssignments(self, index: int, stride: list, card: np.array):
        if (len(stride) == 0):
            return 0
        return [self.getAssignment(index, stride[i], card[i]) for i in range(len(stride))]

    def getOrder(self):
        cliqueSets = [set(cs) for cs in self.cliqueScopes]
        vars = set()
        for cs in cliqueSets:
            vars = vars | cs
        varD = {v: set() for v in vars}
        for v in vars:
            for cs in cliqueSets:
                if v in cs:
                    varD[v] = varD[v] | cs

        vars = np.array([v for v in varD.keys()])
        order = np.argsort([len(varD[vD]) for vD in varD.keys()])
        return vars[order]

    def getStride(self, cliqueScope):
        prod = 1
        a = []
        for i in reversed(cliqueScope):
            a.append(prod)
            prod = prod * self.card[i]
        return list(reversed(a))

    def factorProduct(self, X1, F1, S1, X2, F2, S2):
        j = 0
        k = 0
        clique = np.array(list(set().union(X1, X2)))
        stride = self.getStride(np.array(clique))
        x1i = np.ndarray.flatten(np.array([np.argwhere(clique == i) for i in X1])).astype(np.int64)
        x2i = np.ndarray.flatten(np.array([np.argwhere(clique == i) for i in X2])).astype(np.int64)
        vn = np.product(self.card[clique])
        psi = np.full(vn, 0.0)
        for i in range(np.product(self.card[clique])):
            assign = np.array(self.getAssignments(i, stride, self.card[clique]))
            psi[i] = F1[self.getIndex(assign[x1i], S1)] * F2[self.getIndex(assign[x2i], S2)]
        return clique, psi, stride

    def sumOut(self):
        functions = [(self.cliqueScopes[i], self.functionTables[i], self.stride[i]) for i in
                     range(len(self.cliqueScopes))]
        for o in self.order:
            phi = [(cs, f, s) for cs, f, s in functions if o in cs]
            functions = [(cs, f, s) for cs, f, s in functions if o not in cs]
            newPhi = None
            if len(phi) == 0:
                continue
            elif len(phi) > 1:
                newPhi = self.factorProduct(phi[0][0], phi[0][1], phi[0][2],
                                            phi[1][0], phi[1][1], phi[1][2])
                phi.pop(0)
                phi.pop(0)
                for p in phi:
                    newPhi = self.factorProduct(newPhi[0], newPhi[1], newPhi[2], p[0], p[1], p[2])
            else:
                newPhi = phi[0]
            functions.append(self.sumVariable(newPhi[0], newPhi[1], newPhi[2], o))
        return np.sum(np.log10([f[1][0] for f in functions]))

    def sumVariable(self, X, F, S, v):
        vi = int(np.argwhere(X == v))
        n = np.product(self.card[X])
        newCs = [x for x in X if x != v]
        newPhi = newCs, np.full(n // self.card[v], 0.0), self.getStride(self.card[newCs])
        for i in range(np.product(self.card[X])):
            assignment = self.getAssignments(i, S, self.card[X])
            newPhi[1][self.getIndex(np.delete(assignment, vi), newPhi[2])] += F[i]
        return newPhi

    def instantiateEvidence(self):
        cliqueScopes = [np.array(cs) for cs in self.cliqueScopes]
        for var, val in self.evidence:
            for i, cs in enumerate(cliqueScopes):
                varI = np.argwhere(cs == var)
                if len(varI) != 1:
                    continue
                varI = int(varI)
                newI = []
                for j in range(len(self.functionTables[i])):
                    assignment = self.getAssignments(j, self.stride[i], self.card[cs])
                    if assignment[varI] == val:
                        newI.append(j)
                self.functionTables[i] = self.functionTables[i][newI]
                cliqueScopes[i] = np.delete(cs, varI)
                self.stride[i] = self.getStride(cliqueScopes[i])
        self.cliqueScopes = cliqueScopes

    def parseUAIEvidence(self, evidenceFile: str):
        s = [t for t in open(evidenceFile, "r").read().split(' ') if t]
        observedVariables = int(s.pop(0))
        self.evidence = [(int(s[2 * i]), int(s[2 * i + 1])) for i in range(0, observedVariables)]

    def parseUAI(self, uaiFile: str):
        s = re.sub('^c.*\n?', '', open(uaiFile, "r").read(), flags=re.MULTILINE)
        s = [l for l in s.split('\n') if l]
        s = [l.replace('\t', ' ').replace('\r', ' ').replace('\f', ' ').replace('\v', ' ') for l in s]
        # Below parses the UAI file
        while (len(s) != 0):
            data = s.pop(0)
            if data.upper() in {'MARKOV', 'BAYES'}:
                self.networkType = data
                self.varN = int(s.pop(0))
                self.card = np.array([int(d) for d in s.pop(0).split(' ') if d])
                self.cliques = int(s.pop(0))
                cliqueScopes = []
                while (True):
                    cs = s.pop(0)
                    cliqueScopes += [[int(d) for d in cs.split(' ') if d][1:]]
                    if len([t for t in s[0] if t]) == 1:
                        break
                self.cliqueScopes = cliqueScopes
            if (data.isdigit()):
                entriesN = int(data)
                entries = []
                entriesAdded = 0
                while entriesAdded != entriesN:
                    newEntries = [float(d) for d in s.pop(0).split(' ') if d]
                    entriesAdded += len(newEntries)
                    entries += newEntries
                self.functionTables += [np.array(entries)]
            self.stride = [self.getStride(cs) for cs in self.cliqueScopes]


network = GraphicalModel("1")
print(network.sumOut())
