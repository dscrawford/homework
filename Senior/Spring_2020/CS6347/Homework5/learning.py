import numpy as np
from varElim import GraphicalModel, Network
import sys

dir = './hw5-data/'


# Code made by Brain Khuu: https://stackoverflow.com/questions/3160699/python-progress-bar
# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10  # Modify this to change the length of the progress bar
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "\nHalt...\r\n"
    block = int(round(barLength * progress))
    text = "\r[{0}] {1:.2f}%".format("#" * block + "-" * (barLength - block), progress * 100)
    sys.stdout.write(text)
    sys.stdout.flush()


class Dataset:
    def __init__(self, num_variables, num_rows, data):
        self.num_variables = num_variables
        self.num_rows = num_rows
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __iter__(self):
        self._i = 0
        return iter(self.data)

    def __next__(self):
        if self._i <= self.num_rows:
            self._i += 1
            return self.data[self._i]
        raise StopIteration

    def get_column(self, item):
        return self.data[:, item]

    def get_assignment_count(self, var_assign):
        vars = np.array(list(var_assign.keys()))
        count = 0
        for row in self.data:
            increment = True
            for var in vars:
                if row[var] != var_assign[var]:
                    increment = False
                    break
            count += increment
        return count


class Data_Extractor:
    def __init__(self, file_path):
        self.data = Dataset(*self.parse_file(file_path))

    def get_data(self):
        return self.data

    def parse_file(self, file_path):
        l = list(filter(None, reversed(open(file_path).read().split('\n'))))
        num_variables, num_rows = [int(i) for i in list(filter(None, l.pop().split(' ')))]
        data = np.array([[int(i) for i in list(filter(None, row.split(' ')))] for row in l])
        return num_variables, num_rows, data


class FOD_Learn:
    def __init__(self, file_name, train, verbose=True):
        self.network = Network(file_name, ignore_factors=True)
        self.train = train
        self.verbose = verbose

    def learn_parameters(self, verbose=True):
        if self.verbose:
            print('Learning parameters...')
            update_progress(0)
        for i in range(len(self.network.factors)):
            f = self.network.factors[i]
            cs = f.cliqueScope
            child = cs[-1]
            n = np.product([f.card[c] for c in f.cliqueScope])
            f.functionTable = np.full(n, 0.0)
            for j in range(n):
                assignments = f.getAssignments(j)
                d = {var: assignments[i] for i, var in enumerate(cs)}
                smooth = f.card[child]
                assignment_count = self.train.get_assignment_count(d) + 1
                del (d[child])
                total_count = self.train.get_assignment_count(d) + smooth
                f.functionTable[j] = assignment_count / total_count
            self.network.factors[i].functionTable = f.functionTable
            if self.verbose:
                update_progress((i + 1) / len(self.network.factors))
        if self.verbose:
            print()

    def get_network(self):
        return self.network

    def test_network(self, test, verbose=False):
        if self.verbose:
            print('Computing model predictions')
            update_progress(0)
        pgm = GraphicalModel(self.network)
        ll = []
        for i, t in enumerate(test):
            ll.append(pgm.getLikelihood(evidence=[(j, assign) for j, assign in enumerate(t)]))
            if self.verbose:
                update_progress((i + 1) / len(test))
        return ll


train_data = Data_Extractor(dir + 'dataset1/train-f-1.txt').get_data()
test_data = Data_Extractor(dir + 'dataset1/test.txt').get_data()
a = FOD_Learn(dir + '/dataset1/1', train_data)
a.learn_parameters()
results = a.test_network(test_data)
# print(GraphicalModel(a.get_network()).sumOut())
