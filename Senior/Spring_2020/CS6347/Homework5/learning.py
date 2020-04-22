import numpy as np
from varElim import GraphicalModel, Network, Factor
import sys
from logdouble import Log_Double
from random import uniform, seed
from multiprocessing import Pool

dir = './hw5-data/'


def log_difference(r1, r2):
    return np.sum(np.abs(np.array([f.val for f in r1]) - np.array([f.val for f in r2])))


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


def split_data(data, num_splits):
    n = len(data)
    splitted_data = []
    shift = n // num_splits
    for i in range(num_splits):
        splitted_data.append(data[i * shift:i * shift + shift])
    if n % num_splits != 0:
        splitted_data.append(data[num_splits * shift:num_splits * shift + n % num_splits])
    return splitted_data


def compute_likelihood(completions, network):
    ll = []
    for completion in completions:
        ll.append(network.getLikelihood(evidence={var: d for var, d in enumerate(completion)}))
    return ll


class BigFloat:
    def __init__(self, num):
        self.num = num


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
        data = np.array([[-1 if i == '?' else int(i) for i in list(filter(None, row.split(' ')))] for row in l])
        return num_variables, num_rows, data


class Learner:
    def __init__(self, file_name, train, verbose=True, ignore_factors=True, num_processes=1):
        self.file_name = file_name
        self.network = Network(file_name, ignore_factors=ignore_factors)
        self.train = train
        self.verbose = verbose
        self.num_processes = num_processes

    def get_network(self):
        return self.network

    def learn_parameters(self):
        pass

    def test_network_fully_observed(self, test):
        if self.verbose:
            print('Computing model predictions using network structure from ' + self.file_name)
            update_progress(0)
        network = GraphicalModel(self.network)
        ll = []
        for i,data_set in enumerate(split_data(test, 10)):
            p = Pool(self.num_processes)
            ll_set = p.starmap(compute_likelihood, [(data, network) for data in split_data(data_set, self.num_processes)])
            ll_set = [likelihood for L in ll_set for likelihood in L]
            for likelihood in ll_set:
                ll.append(likelihood)
            if self.verbose:
                update_progress((i + 1) / 10)
        if self.verbose:
            print()
        return np.array(ll)


class Trained_Learn(Learner):
    def __init__(self, file_name, train=None, verbose=True, num_processes=1):
        Learner.__init__(self, file_name, train, verbose, ignore_factors=False, num_processes=num_processes)

    def get_network(self):
        return Learner.get_network(self)

    def learn_parameters(self):
        print('Parameters already learned in pre-trained model')
        pass

    def test_network_fully_observed(self, test):
        return Learner.test_network_fully_observed(self, test)


class FOD_Learn(Learner):
    def __init__(self, file_name, train, verbose=True, num_processes=1):
        Learner.__init__(self, file_name, train, verbose, ignore_factors=False, num_processes=num_processes)

    def learn_parameters(self):
        if self.verbose:
            print('Learning parameters on file ' + self.file_name + ' using MLE approach.')
            update_progress(0)
        for i in range(len(self.network.factors)):
            f = self.network.factors[i]
            cs = f.cliqueScope
            child = cs[-1]
            n = np.product([f.card[c] for c in f.cliqueScope])
            f.functionTable = np.full(n, Log_Double())
            for j in range(n):
                assignments = f.getAssignments(j)
                d = {var: assignments[i] for i, var in enumerate(cs)}
                smooth = f.card[child]
                assignment_count = self.train.get_assignment_count(d) + 1
                del (d[child])
                total_count = self.train.get_assignment_count(d) + smooth
                f.functionTable[j] = Log_Double(assignment_count / total_count)
            self.network.factors[i].functionTable = f.functionTable
            if self.verbose:
                update_progress((i + 1) / len(self.network.factors))
        if self.verbose:
            print()

    def get_network(self):
        return Learner.get_network(self)

    def test_network_fully_observed(self, test):
        return Learner.test_network_fully_observed(self, test)


class POD_EM_Learn(Learner):
    def __init__(self, file_name, train, verbose=True, stabilizer=Log_Double(1e-100), dropout=None, num_processes=1):
        stabilizer = Log_Double(stabilizer)
        Learner.__init__(self, file_name, train, verbose, ignore_factors=True,num_processes=num_processes)
        self.stabilizer = stabilizer
        self.dropout = Log_Double(dropout)
        self.verbose = verbose
        self.num_processes = num_processes

    def get_network(self):
        return Learner.get_network(self)

    def compute_ess(self, completions, counts=None):
        n = len(completions)
        if counts is None:
            counts = np.full(n, 1)
        M = [np.full(len(f.functionTable), Log_Double()) for f in self.network.factors]
        p = Pool(self.num_processes)
        network = GraphicalModel(self.network)
        likelihood = p.starmap(compute_likelihood,
                               [(data, network) for data in split_data(completions, self.num_processes)])
        likelihood = [l for L in likelihood for l in L]
        for i in range(n):
            if self.dropout is not None and likelihood[i] > self.dropout:
                continue
            for m in range(len(M)):
                assignment = completions[i][self.network.factors[m].cliqueScope]
                index = self.network.factors[m].getIndex(assignment)
                M[m][index] += Log_Double(counts[i]) * likelihood[i]
        return M

    def generate_all_data_completions(self, data):
        l = []
        for row in data:
            r = row.copy()
            unknown_var_i = []
            for i in range(len(row)):
                if r[i] == -1:
                    unknown_var_i.append(i)
            completion_generator = Factor(unknown_var_i, None, {i: self.network.card[i] for i in unknown_var_i})
            for i in range(completion_generator.getSize()):
                r[unknown_var_i] = completion_generator.getAssignments(i)
                l.append(r.copy())
        l = np.array(l)
        return np.unique(l, return_counts=True, axis=0)

    def random_gen_parameters(self):
        M = [np.array([Log_Double(uniform(0, 1)) for _ in range(len(f.functionTable))]) for f in self.network.factors]
        return self.normalize_network(M)

    def set_parameters(self, M):
        self.network.factors = [Factor(self.network.factors[i].cliqueScope, M[i], self.network.factors[i].card)
                                for i in range(len(self.network.factors))]

    def stabilize(self, value, stabilizer=None):
        if stabilizer is None:
            stabilizer = self.stabilizer
        if value.is_zero or value < stabilizer:
            return Log_Double(stabilizer)
        return value

    def normalize_network(self, M):
        for i, f in enumerate(self.network.factors):
            child_card = f.card[f.cliqueScope[-1]]
            for j in range(len(f.functionTable) // child_card):
                total_sum = sum([M[i][child_card * j + k] for k in range(child_card)])
                if total_sum.is_zero:
                    for k in range(child_card):
                        M[i][child_card * j + k] = f.functionTable[child_card * j + k]
                else:
                    for k in range(child_card):
                        M[i][child_card * j + k] = M[i][child_card * j + k] / total_sum
        return M

    def learn_parameters(self, num_iterations=20):
        self.set_parameters(self.random_gen_parameters())
        if self.verbose:
            print('Running EM algorithm to determine parameters for PGM in ' + self.file_name)
            update_progress(0)
        completions, counts = self.generate_all_data_completions(self.train)
        for t in range(num_iterations):
            # E-step
            M = self.normalize_network(self.compute_ess(completions, counts))
            M = self.normalize_network([[self.stabilize(lik) for lik in F] for F in M])
            # M-step
            self.set_parameters(M)
            if self.verbose:
                update_progress((t + 1) / num_iterations)
        print([f.get_float() for f in self.network.factors[0].functionTable])
        if self.verbose:
            print()


seed(123)
dataset_path = dir + 'dataset2/'
train_data = Data_Extractor(dataset_path + 'train-p-4.txt').get_data()
test_data = Data_Extractor(dataset_path + 'test.txt').get_data()
b = Trained_Learn(dataset_path + '2', num_processes=8)
d = POD_EM_Learn(dataset_path + '2', train_data, stabilizer=1e-2, verbose=True, dropout=1e-10, num_processes=8)
d.learn_parameters(num_iterations=10)
pretrained_likelihoods = b.test_network_fully_observed(test_data)
trained_likelihoods = d.test_network_fully_observed(test_data)
print(log_difference(pretrained_likelihoods, trained_likelihoods) / len(test_data))
