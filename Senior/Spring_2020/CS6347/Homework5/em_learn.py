import numpy as np
from learner import Learner, update_progress
from logdouble import Log_Double
from varElim import GraphicalModel, Network, Factor
from random import uniform, shuffle, sample

class EM_Learn(Learner):
    def __init__(self, file_name=None, verbose=True, stabilizer=Log_Double(1e-100), dropout=None, num_processes=1,
                 network=None):
        if file_name is None and network is None:
            print('Error: No way to initialize network')
        stabilizer = Log_Double(stabilizer)
        if network is None:
            Learner.__init__(self, file_name, verbose, ignore_factors=True, num_processes=num_processes)
        else:
            self.network = network
            self.card = self.network.card
            Learner.__init__(self, file_name=None, verbose=verbose,
                             ignore_factors=True, num_processes=num_processes)

        self.stabilizer = stabilizer
        if dropout is not None:
            self.dropout = Log_Double(dropout)
        else:
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
        total_count = 0
        for i in range(n):
            total += likelihoods[i]
            total_count += 1
            if self.dropout is not None and self.dropout < likelihoods[i]:
                continue
            for m in range(len(M)):
                assignment = completions[i][self.network.factors[m].cliqueScope]
                index = self.network.factors[m].getIndex(assignment)
                M[m][index] += Log_Double(counts[i]) * likelihoods[i]
        return M, total

    def compute_likelihoods(self, data):
        ll = []
        for completion in data:
            ll.append(self.pgm.getLikelihood(evidence={var: d for var, d in enumerate(completion)}))
        return ll

    def compute_likelihood(self, data):
        return self.pgm.getLikelihood(evidence={var: d for var, d in enumerate(data)})

    def generate_all_data_completions(self, data):
        l = []
        for row in data:
            r = row.copy()
            unknown_var_i = []
            for i in range(len(row)):
                if r[i] == -1:
                    unknown_var_i.append(i)
            completion_generator = Factor(unknown_var_i, None, {i: self.card[i] for i in unknown_var_i})
            for i in range(completion_generator.getSize()):
                r[unknown_var_i] = completion_generator.getAssignments(i)
                l.append(r.copy())
        l = np.array(l)
        return np.unique(l, return_counts=True, axis=0)

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
                if total_sum.is_zero:
                    for k in range(child_card):
                        M[i][child_card * j + k] = f.functionTable[child_card * j + k]
                else:
                    for k in range(child_card):
                        M[i][child_card * j + k] = M[i][child_card * j + k] / total_sum + self.stabilizer
                    total_sum = sum([M[i][child_card * j + k] for k in range(child_card)])
                    for k in range(child_card):
                        M[i][child_card * j + k] = (M[i][child_card * j + k] / total_sum)
        return M


class POD_EM_Learn(EM_Learn):
    def __init__(self, file_name, verbose=True, stabilizer=Log_Double(1e-5), dropout=None, num_processes=1):
        EM_Learn.__init__(self, file_name, verbose, stabilizer, dropout, num_processes)

    def learn_parameters(self, train, num_iterations=20):
        self.set_parameters(self.random_gen_parameters())
        self.pgm = GraphicalModel(self.network)
        if self.verbose:
            print('Running EM algorithm to determine parameters for PGM in ' + self.file_name)
            update_progress(0)
        completions, counts = self.generate_all_data_completions(train)
        for t in range(num_iterations):
            # E-step
            M, _ = self.compute_ess(completions, counts)
            M = self.normalize_network(M)
            # M-step
            self.set_parameters(M)
            if self.verbose:
                update_progress((t + 1) / num_iterations)
        if self.verbose:
            print()


class Mixture_Random_Bayes_Learn(EM_Learn):
    def __init__(self, num_vars, verbose=True, stabilizer=Log_Double(1e-5), dropout=None, num_processes=1, k=2):
        Learner.__init__(self, file_name=None, verbose=verbose, ignore_factors=True,
                         num_processes=num_processes)
        stabilizer = Log_Double(stabilizer)
        self.stabilizer = Log_Double(stabilizer)
        self.verbose = verbose
        self.card = {var: 2 for var in range(num_vars)}
        self.networks = self.generate_k_dags(num_vars, k)
        self.learners = [EM_Learn(file_name=None, network=network, num_processes=num_processes, dropout=dropout,
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

    def learn_parameters(self, train, num_iterations=20):
        if self.verbose:
            print('Training', self.k, 'randomly created DAGs with EM')
            update_progress(0)
        completions, counts = self.generate_all_data_completions(train)
        for learner in self.learners:
            learner.set_parameters(learner.random_gen_parameters())
        for t in range(num_iterations):
            learner_likelihoods = []
            for learner in self.learners:
                M, learner_likelihood = learner.compute_ess(completions, counts)
                M = learner.normalize_network(M)
                learner_likelihoods.append(learner_likelihood)
                learner.set_parameters(M)
            self.p = self.normalize_p([learner_likelihoods[i] for i in range(self.k)])
            if self.verbose:
                update_progress((t + 1) / num_iterations)
        if self.verbose:
            print()