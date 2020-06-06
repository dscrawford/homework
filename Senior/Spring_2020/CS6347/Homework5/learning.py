from random import seed
from Data_Tools import Data_Extractor
from learner import Trained_Learn, log_difference
from fod_learn import FOD_Learn
from em_learn import POD_EM_Learn, Mixture_Random_Bayes_Learn
from numpy import array, sqrt
from sys import exit
import argparse

parser = argparse.ArgumentParser(description='Trains Bayesian networks using different models')
parser.add_argument('input_uai_file', metavar='input_uai_file', type=str, help='File containing structure of PGM')
parser.add_argument('task_id', metavar='task_id', type=int, help='Task which to perform', choices=[1,2,3])
parser.add_argument('training_data_file', metavar='training_data_file', type=str, help='Training data for PGM')
parser.add_argument('testing_data_file', metavar='testing_data_file', type=str, help='Testing data for PGM')
parser.add_argument('--k', metavar='k', type=int, help='Constant for increasing number of DAGs in Mixture Bayes model',
                    default=2)
parser.add_argument('--processes', metavar='num_processes', type=int, help='Number of processes to run',
                    default=1)
parser.add_argument('--no_verbose', action='store_false', help='Boolean on if verbose should be disabled',
                    default=False)

args = parser.parse_args()
input_uai_file = args.input_uai_file
task_id = args.task_id
training_data_file = args.training_data_file
testing_data_file = args.testing_data_file
k = args.k
num_processes = args.processes
verbose = not args.no_verbose

train_data = Data_Extractor(training_data_file).get_data()
test_data = Data_Extractor(testing_data_file).get_data()
true = Trained_Learn(input_uai_file, num_processes=num_processes, verbose=verbose)
print('Computing true likelihoods for data in ' + testing_data_file)
true_likelihoods = true.get_likelihoods(test_data)
if task_id == 1:
    pred = FOD_Learn(input_uai_file, num_processes=num_processes, verbose=verbose)
    pred.learn_parameters(train_data)
elif task_id == 2:
    pred = POD_EM_Learn(input_uai_file, num_processes=8, verbose=verbose)
    pred.learn_parameters(train_data, num_iterations=20)
elif task_id == 3:
    pred = Mixture_Random_Bayes_Learn(len(train_data[0]), k=k, num_processes=8, verbose=verbose)
    pred.learn_parameters(train_data, num_iterations=20)
else:
    print('Error: should not reach here')
    exit(1)

print('Computing predicted likelihoods for data in  ' + testing_data_file)
pred_likelihoods = pred.get_likelihoods(test_data)
print('---------------------------\n',
      'log likelihood difference =', log_difference(true_likelihoods, pred_likelihoods),
      '\n---------------------------')

# Below is old code used for computing all data sequentially
# dataset_no = '1'
# dir = './hw5-data/'
#
# k = 1
#
# def print_mixture_results(dir, dataset_no, model, isPartial=False, k=k):
#     dataset_no = str(dataset_no)
#     if isPartial:
#         train_choice = 'p'
#     else:
#         train_choice = 'f'
#     dataset_path = dir + 'dataset' + dataset_no + '/'
#     datasets = [dataset_path + 'train-' + train_choice + '-1.txt',
#                 dataset_path + 'train-' + train_choice + '-2.txt',
#                 dataset_path + 'train-' + train_choice + '-3.txt',
#                 dataset_path + 'train-' + train_choice + '-4.txt']
#     test_data = Data_Extractor(dataset_path + 'test.txt').get_data()
#     trained = Trained_Learn(dataset_path + dataset_no, num_processes=8)
#     true_likelihood = trained.test_network_fully_observed(test_data)
#     for dataset in datasets:
#         print('Currently on dataset: ', dataset)
#         log_diffs = []
#         for i in range(5):
#             print('Iteration: ', i)
#             train_data = Data_Extractor(dataset).get_data()
#             predictor = model(len(train_data[0]), k=k, num_processes=8, verbose=False)
#             predictor.learn_parameters(train_data)
#             pred_likelihood = predictor.test_network_fully_observed(test_data)
#             log_diffs.append(log_difference(true_likelihood, pred_likelihood))
#             print('Diff average: ', log_diffs[i] / len(test_data))
#         log_diffs = array(log_diffs)
#         average_log_diff = sum(log_diffs) / 5
#         sd_log_diff = sqrt(sum((log_diffs - average_log_diff) ** 2) / 5)
#         print('Average: ', average_log_diff)
#         print('Standard Deviation: ', sd_log_diff)
# print('k=2')
# k = 2
# print_results(dir, '2', Mixture_Random_Bayes_Learn, isPartial=False)
# print('k=4')
# k = 4
# print_results(dir, '2', Mixture_Random_Bayes_Learn, isPartial=False)
# print('k=6')
# k = 6
# print_results(dir, '2', Mixture_Random_Bayes_Learn, isPartial=False)
