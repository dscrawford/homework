import numpy as np
import pandas as pd
import regex as re


class GraphicalModel:
    networkType = ""
    varN = 0
    varCardinalities = np.array([])
    cliques = 0
    cliqueScopes = np.array([])
    functionTables = []
    evidence = np.array([])

    def __init__(self, uaiFile: str):
        self.parseUAI(uaiFile + ".uai")
        self.parseUAIEvidence(uaiFile + ".uai.evid")

    def instantiateEvidence(self):
        evidenceVars = list([i[0] for i in self.evidence])
        evidenceVals = list([i[1] for i in self.evidence])
        card         = self.varCardinalities
        for csi, cs in enumerate(self.cliqueScopes):
            M = np.reshape(self.functionTables[csi], self.varCardinalities[self.cliqueScopes[csi]])


    def parseUAIEvidence(self, evidenceFile: str):
        s = [t for t in open(evidenceFile, "r").read().split(' ') if t]
        observedVariables = int(s.pop(0))
        self.evidence = [(int(s[2*i]),int(s[2*i + 1])) for i in range(0, observedVariables)]

    def parseUAI(self, uaiFile: str):
        s = re.sub('^c.*\n?', '', open(uaiFile, "r").read(), flags=re.MULTILINE)
        s = [l for l in s.split('\n') if l]
        # Below parses the UAI file
        while (len(s) != 0):
            data = s.pop(0)
            if (data.upper() in ['MARKOV', 'BAYES']):
                self.networkType = data
                self.varN = int(s.pop(0))
                self.varCardinalities = np.array([int(d) for d in s.pop(0).split(' ') if d])
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


network = GraphicalModel("network")