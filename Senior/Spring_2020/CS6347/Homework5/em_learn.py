import numpy as np
from learner import Learner, update_progress, split_data
from fod_learn import FOD_Learn
from logdouble import Log_Double
from varElim import GraphicalModel, Network, Factor
from random import uniform, shuffle, sample
from multiprocessing import Pool
from time import time


class EM_Learn(Learner):
    def __init__(self, file_name=None, verbose=True, stabilizer=Log_Double(1e-100), num_processes=1,
                 network=None):
        if file_name is None and network is None:
            print('Error: No way to initialize network')
        stabilizer = Log_Double(stabilizer)
        if network is None:
            Learner.__init__(self, file_name, verbose, ignore_factors=True, num_processes=num_processes)
        else:
            Learner.__init__(self, file_name=None, verbose=verbose,
                             ignore_factors=True, num_processes=num_processes, network=network)
        self.stabilizer = stabilizer
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
        likelihoods = self.get_all_normalized_weights(completions)
        for i in range(n):
            for j in range(len(completions[i])):
                for m in range(len(M)):
                    assignment = completions[i][j][self.network.factors[m].cliqueScope]
                    index = self.network.factors[m].getIndex(assignment)
                    M[m][index] += likelihoods[i][j] * Log_Double(counts[i])
        return M

    def compute_likelihoods(self, data):
        ll = []
        for completion in data:
            ll.append(self.pgm.getLikelihood(evidence={var: d for var, d in enumerate(completion)}))
        return ll

    def compute_likelihood(self, data):
        return self.pgm.getLikelihood(evidence={var: d for var, d in enumerate(data)})

    def get_normalized_weights(self, data_segment):
        ll = self.compute_likelihoods(data_segment)
        total_sum = sum(ll)
        return [l / total_sum for l in ll]

    def get_all_normalized_weights(self, data):
        p = Pool(self.num_processes)
        self.get_normalized_weights(data[0])
        weights = p.map(self.get_normalized_weights, data)
        p.terminate()
        return weights

    def generate_all_data_completions(self, data):
        l = []
        reduced_data, counts = np.unique(data.data, return_counts=True, axis=0)
        for row in reduced_data:
            r = row.copy()
            unknown_var_i = []
            for i in range(len(row)):
                if r[i] == -1:
                    unknown_var_i.append(i)
            completion_generator = Factor(unknown_var_i, None, {i: self.card[i] for i in unknown_var_i})
            new_l = []
            for i in range(completion_generator.getSize()):
                r[unknown_var_i] = completion_generator.getAssignments(i)
                new_l.append(r.copy())
            l.append(new_l)
        return np.array(l), counts

    def random_gen_parameters(self):
        M = [np.array([Log_Double(uniform(0, 1)) for _ in range(len(f.functionTable))]) for f in self.network.factors]
        return self.normalize_network(M)

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
    def __init__(self, file_name, verbose=True, stabilizer=Log_Double(1e-5), num_processes=1, network=None):
        EM_Learn.__init__(self, file_name, verbose, stabilizer, num_processes, network)

    def learn_parameters(self, train, num_iterations=20):
        self.set_parameters(self.random_gen_parameters())
        self.pgm = GraphicalModel(self.network)
        if self.verbose:
            print('Running EM algorithm to determine parameters for PGM in ' + self.file_name)
            update_progress(0)
        completions, counts = self.generate_all_data_completions(train)
        for t in range(num_iterations):
            # E-step
            M = self.normalize_network(self.compute_ess(completions, counts))
            # M-step
            self.set_parameters(M)
            if self.verbose:
                update_progress((t + 1) / num_iterations)
        if self.verbose:
            print()


class Mixture_Random_Bayes_Learn(EM_Learn):
    def __init__(self, num_vars, verbose=True, stabilizer=Log_Double(1e-5), num_processes=1, k=2):
        self.num_processes = num_processes
        stabilizer = Log_Double(stabilizer)
        self.stabilizer = Log_Double(stabilizer)
        self.verbose = verbose
        self.card = {var: 2 for var in range(num_vars)}
        self.networks = self.generate_k_dags(num_vars, k)
        self.card[num_vars] = k
        self.learners = [FOD_Learn(file_name=None, network=network, num_processes=num_processes,
                                   verbose=False) for network in self.networks]
        self.k = k
        p = [uniform(0, 1) for ki in range(k)]
        self.p = self.normalize_p(p)
        self.file_name = 'randomly generated DAGs'

    def normalize_p(self, p):
        p_sum = Log_Double(sum(p))
        p = [(Log_Double(pi) / p_sum) + self.stabilizer for pi in p]
        p_sum = Log_Double(sum(p))
        return [Log_Double(pi) / p_sum for pi in p]

    def generate_k_dags(self, variable_count, k):
        networkType = "BAYES"
        n = variable_count
        varN = n
        cliques = n
        card = self.card
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
                parents = sample(processed_nodes, num_parents)
                cliqueScope = parents + [node]
                new_card = {var: card[var] for var in cliqueScope}
                new_factors.append(Factor(cliqueScope, None, new_card))
                processed_nodes.append(node)
            networks[i].factors = new_factors
        return networks

    def compute_likelihood(self, data):
        likelihood = []
        for learner in self.learners:
            likelihood.append(learner.compute_likelihood(data))
        return np.sum([likelihood[k] * self.p[k] for k in range(self.k)])

    def compute_likelihoods(self, data):
        return [self.compute_likelihood(row) for row in data]

    def get_p_normalized_weight(self, row):
        weight = []
        for i, learner in enumerate(self.learners):
            weight.append(self.p[i] * learner.compute_likelihood(row[i]))
        weight_sum = sum(weight)
        return [w / weight_sum for w in weight]

    def get_p_normalized_weights(self, data, counts):
        p = Pool(self.num_processes)
        weights = p.map(self.get_p_normalized_weight, data)
        p.terminate()
        weights = [[Log_Double(counts[i]) * weights[i][j] for j in range(len(weights[i]))] for i in range(len(counts))]
        return weights

    def set_parameters(self, weights):
        p = [Log_Double() for _ in range(self.k)]
        for i in range(self.k):
            for j in range(len(weights)):
                p[i] += weights[j][i]
        self.p = self.normalize_p(p)

    def learn_parameters(self, train, num_iterations=20):
        if self.verbose:
            print('Training', self.k, 'randomly created DAGs with EM')
            update_progress(0)
        for learner in self.learners:
            learner.learn_parameters(train)
        train = np.array([list(tr) + [-1] for tr in train])
        completions, counts = self.generate_all_data_completions(train)
        for t in range(num_iterations):
            self.set_parameters(self.get_p_normalized_weights(completions, counts))
            if self.verbose:
                update_progress((t + 1) / num_iterations)
        if self.verbose:
            print()
