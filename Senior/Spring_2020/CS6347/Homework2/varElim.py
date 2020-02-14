import numpy as np
import pandas as pd
import regex as re

s = re.sub('^c.*\n?', '', open("network.uai", "r").read(), flags=re.MULTILINE)
s = [l for l in s.split('\n') if l]


networkType = s.pop(0)
varN        = int(s.pop(0))
varCard     = [int(d) for d in s.pop(0).split(' ')]
cliques     = int(s.pop(0))

cliqueScopes = [[int(d) for d in s[i].split(' ') if d] for i in range(4,cliqueI + 1)]

functionTables = []
