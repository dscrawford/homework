import numpy as np
from varElim import GraphicalModel, Network, Factor
import sys
from logdouble import Log_Double
from random import uniform, shuffle, seed
from multiprocessing.pool import ThreadPool as Pool

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


# Need to speed up this operation a lot
def split_data(data, num_splits):
    n = len(data)
    shift = n // num_splits
    splitted_data = [data[i * shift:i * shift + shift] for i in range(num_splits)]
    if n % num_splits != 0:
        splitted_data.append(data[num_splits * shift:num_splits * shift + n % num_splits])
    return splitted_data


def compute_likelihoods(completions, pgm):
    ll = []
    for completion in completions:
        ll.append(pgm.getLikelihood(evidence={var: d for var, d in enumerate(completion)}))
    return ll


def compute_likelihood(completion, pgm):
    return pgm.getLikelihood(evidence={var: d for var, d in enumerate(completion)})


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
    def __init__(self, file_name, verbose=True, ignore_factors=True, num_processes=1):
        self.file_name = file_name
        self.network = Network(file_name, ignore_factors=ignore_factors)
        self.verbose = verbose
        self.num_processes = num_processes

    def get_network(self):
        return self.network

    def learn_parameters(self, train):
        pass

    def test_network_fully_observed(self, test):
        if self.verbose:
            print('Computing model predictions using network structure from ' + self.file_name)
            update_progress(0)
        network = GraphicalModel(self.network)
        ll = []
        for i, data_set in enumerate(split_data(test, 10)):
            p = Pool(self.num_processes)
            ll_set = p.starmap(compute_likelihoods,
                               [(data, network) for data in split_data(data_set, self.num_processes)])
            ll_set = [likelihood for L in ll_set for likelihood in L]
            for likelihood in ll_set:
                ll.append(likelihood)
            if self.verbose:
                update_progress((i + 1) / 10)
        if self.verbose:
            print()
        return np.array(ll)


class Trained_Learn(Learner):
    def __init__(self, file_name, verbose=True, num_processes=1):
        Learner.__init__(self, file_name, verbose, ignore_factors=False, num_processes=num_processes)

    def get_network(self):
        return Learner.get_network(self)

    def learn_parameters(self, train):
        print('Parameters already learned in pre-trained model')
        pass

    def test_network_fully_observed(self, test):
        return Learner.test_network_fully_observed(self, test)


class FOD_Learn(Learner):
    def __init__(self, file_name, verbose=True, num_processes=1):
        Learner.__init__(self, file_name, verbose, ignore_factors=True, num_processes=num_processes)

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
            for j in range(n):
                assignments = f.getAssignments(j)
                d = {var: assignments[i] for i, var in enumerate(cs)}
                smooth = f.card[child]
                assignment_count = train.get_assignment_count(d) + 1
                del (d[child])
                total_count = train.get_assignment_count(d) + smooth
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


class EM_Learn(Learner):
    def __init__(self, file_name=None, verbose=True, stabilizer=Log_Double(1e-100), dropout=None, num_processes=1,
                 network=None):
        if file_name is None and network is None:
            print('Error: No way to initialize network')
        stabilizer = Log_Double(stabilizer)
        if network is None:
            Learner.__init__(self, file_name, verbose, ignore_factors=True, num_processes=num_processes)
        else:
            Learner.__init__(self, file_name=None, verbose=verbose,
                             ignore_factors=True, num_processes=num_processes)
            self.network = network
        self.stabilizer = stabilizer
        if dropout is not None:
            self.dropout = Log_Double(dropout)
        self.dropout = None
        self.verbose = verbose
        self.num_processes = num_processes
        self.pgm = GraphicalModel(self.network)

    def get_network(self):
        return Learner.get_network(self)

    def compute_ess(self, completions, counts=None):
        n = len(completions)
        if counts is None:
            counts = np.full(n, 1)
        M = [np.full(len(f.functionTable), Log_Double()) for f in self.network.factors]
        likelihoods = self.get_likelihoods(completions)
        total = Log_Double()
        for i in range(n):
            for j in range(len(completions[i])):
                total += likelihoods[i][j]
                if self.dropout is not None:
                    if self.dropout < likelihoods[i][j]:
                        continue
                for m in range(len(M)):
                    assignment = completions[i][j][self.network.factors[m].cliqueScope]
                    index = self.network.factors[m].getIndex(assignment)
                    M[m][index] += Log_Double(counts[i][j]) * likelihoods[i][j]
        return M

    def compute_likelihoods(self, data):
        ll = []
        for completion in data:
            ll.append(self.pgm.getLikelihood(evidence={var: d for var, d in enumerate(completion)}))
        return ll

    def compute_likelihood(self, data):
        return self.pgm.getLikelihood(evidence={var: d for var, d in enumerate(data)})

    def get_likelihoods(self, completions):
        p = Pool(self.num_processes)
        likelihoods = p.map(self.compute_likelihoods, completions)
        p.terminate()
        return likelihoods

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
                for k in range(child_card):
                    M[i][child_card * j + k] = (M[i][child_card * j + k] / total_sum) + self.stabilizer
                total_sum = sum([M[i][child_card * j + k] for k in range(child_card)])
                for k in range(child_card):
                    M[i][child_card * j + k] = (M[i][child_card * j + k] / total_sum)
        return M


class POD_EM_Learn(EM_Learn):
    def __init__(self, file_name, verbose=True, stabilizer=Log_Double(1e-100), dropout=None, num_processes=1):
        EM_Learn.__init__(self, file_name, verbose, stabilizer, dropout, num_processes)

    def learn_parameters(self, train, num_iterations=20):
        self.set_parameters(self.random_gen_parameters())
        self.pgm = GraphicalModel(self.network)
        if self.verbose:
            print('Running EM algorithm to determine parameters for PGM in ' + self.file_name)
            update_progress(0)
        completions, counts = self.generate_all_data_completions(train)
        completions = split_data(completions, self.num_processes)
        counts = split_data(counts, self.num_processes)
        for t in range(num_iterations):
            # E-step
            M = self.normalize_network(self.compute_ess(completions, counts))
            # M-step
            self.set_parameters(M)
            if self.verbose:
                update_progress((t + 1) / num_iterations)
        print([f.get_float() for f in self.network.factors[0].functionTable])
        if self.verbose:
            print()


class Mixture_Random_Bayes_Learn(EM_Learn):
    def __init__(self, train, verbose=True, stabilizer=Log_Double(1e-100), dropout=None, num_processes=1, k=2):
        Learner.__init__(self, file_name=None, verbose=verbose, ignore_factors=True,
                         num_processes=num_processes)
        stabilizer = Log_Double(stabilizer)
        self.stabilizer = stabilizer
        self.verbose = verbose
        self.networks = self.generate_k_dags(len(train[0]), k)
        self.learners = EM_Learn()

    def generate_k_dags(self, variable_count, k):
        networkType = "BAYES"
        n = variable_count
        varN = n
        cliques = n
        card = {var: 2 for var in range(n)}
        networks = np.array([Network(file_name=None).create_network(networkType, varN, cliques, [], card)
                             for _ in range(k)])
        for i in range(k):
            nodes = list(range(n))
            processed_nodes = []
            shuffle(nodes)
            root = nodes.pop()
            new_factors = [Factor([root], np.full(card[root], 0), {root: card[root]})]
            processed_nodes.append(root)
            while len(nodes) != 0:
                node = nodes.pop()
                num_parents = int(uniform(0, 4)) % len(processed_nodes)
                parents = []
                for _ in range(num_parents):
                    parents.append(processed_nodes[int(uniform(0, len(processed_nodes)))])
                cliqueScope = parents + [node]
                new_card = {var: card[var] for var in cliqueScope}
                new_factors.append(Factor(cliqueScope,
                                          np.full(int(np.product([new_card[var] for var in cliqueScope])), 0),
                                          new_card))
                processed_nodes.append(node)
            networks[i].factors = new_factors
        return networks

    def learn_parameters(self, train):
        completions, counts = self.generate_all_data_completions(train)


dataset_path = dir + 'dataset2/'
train_data = Data_Extractor(dataset_path + 'train-p-2.txt').get_data()
test_data = Data_Extractor(dataset_path + 'test.txt').get_data()[0:100]
b = Trained_Learn(dataset_path + '2', num_processes=8)
d = POD_EM_Learn(dataset_path + '2', num_processes=8, dropout=None, stabilizer=1e-5)
d.learn_parameters(train_data, num_iterations=20)
pretrained_likelihoods = b.test_network_fully_observed(test_data)
trained_likelihoods = d.test_network_fully_observed(test_data)
print(log_difference(pretrained_likelihoods, trained_likelihoods) / len(test_data))
