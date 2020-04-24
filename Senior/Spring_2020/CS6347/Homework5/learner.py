import numpy as np
import sys
from varElim import GraphicalModel, Network, Factor
from multiprocessing import Pool


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


class Learner:
    def __init__(self, file_name, verbose=True, ignore_factors=True, num_processes=1):
        self.file_name = file_name
        if file_name is not None:
            self.network = Network(file_name, ignore_factors=ignore_factors)
            self.pgm = GraphicalModel(self.network)
            self.card = self.network.card
        self.verbose = verbose
        self.num_processes = num_processes

    def get_network(self):
        return self.network

    def learn_parameters(self, train):
        pass

    def compute_likelihoods(self, data):
        ll = []
        for completion in data:
            ll.append(self.pgm.getLikelihood(evidence={var: d for var, d in enumerate(completion)}))
        return ll

    def compute_likelihood(self, data):
        return self.pgm.getLikelihood(evidence={var: d for var, d in enumerate(data)})

    def set_parameters(self, M):
        self.network.factors = [Factor(self.network.factors[i].cliqueScope, M[i], self.network.factors[i].card)
                                for i in range(len(self.network.factors))]
        self.pgm = GraphicalModel(self.network)

    def get_likelihoods(self, completions):
        p = Pool(self.num_processes)
        likelihoods = [l for l in p.map(self.compute_likelihood, completions)]
        p.terminate()
        return likelihoods

    def test_network_fully_observed(self, test):
        if self.verbose:
            print('Computing model predictions using network structure from ' + self.file_name)
            update_progress(0)
        ll = []
        for i, data_set in enumerate(split_data(test, 10)):
            ll_set = self.get_likelihoods(data_set)
            ll = ll + ll_set
            if self.verbose:
                update_progress((i + 1) / 10)
        if self.verbose:
            print()
        return np.array(ll)


class Trained_Learn(Learner):
    def __init__(self, file_name, verbose=True, num_processes=1):
        Learner.__init__(self, file_name, verbose, ignore_factors=False, num_processes=num_processes)

    def get_network(self):
        return self.get_network()

    def learn_parameters(self, train):
        print('Parameters already learned in pre-trained model')
        pass

    def test_network_fully_observed(self, test):
        return Learner.test_network_fully_observed(self, test)
