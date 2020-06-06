import numpy as np
from learner import Learner, update_progress
from varElim import GraphicalModel
from logdouble import Log_Double

class FOD_Learn(Learner):
    def __init__(self, file_name, verbose=True, num_processes=1, network=None):
        Learner.__init__(self, file_name, verbose, ignore_factors=True, num_processes=num_processes,network=network)

    def learn_parameters(self, train):
        if self.verbose:
            print('Learning parameters on file ' + self.file_name + ' using MLE approach.')
            update_progress(0)
        for i in range(len(self.network.factors)):
            f = self.network.factors[i]
            cs = f.cliqueScope
            child = cs[-1]
            n = np.product([f.card[c] for c in f.cliqueScope])
            f.functionTable = np.full(n, Log_Double())
            smooth = f.card[child]
            for j in range(n):
                assignments = f.getAssignments(j)
                d = {var: assignments[i] for i, var in enumerate(cs)}
                assignment_count = train.get_assignment_count(d) + 1
                del (d[child])
                total_count = train.get_assignment_count(d) + smooth
                f.functionTable[j] = Log_Double(assignment_count / total_count)
            self.network.factors[i].functionTable = f.functionTable
            if self.verbose:
                update_progress((i + 1) / len(self.network.factors))
        self.pgm = GraphicalModel(self.network)
        if self.verbose:
            print()

    def get_network(self):
        return Learner.get_network(self)

    def test_network_fully_observed(self, test):
        return Learner.test_network_fully_observed(self, test)