from random import seed
from Data_Tools import Data_Extractor
from learner import Trained_Learn, log_difference
from fod_learn import FOD_Learn
from em_learn import POD_EM_Learn, Mixture_Random_Bayes_Learn

dir = './hw5-data/'


dataset_path = dir + 'dataset2/'
test_data = Data_Extractor(dataset_path + 'test.txt').get_data()[0:100]
train_data = Data_Extractor(dataset_path + 'train-p-' + str(2) + ".txt").get_data()
a = Trained_Learn(dataset_path + '2', num_processes=8)
b = Mixture_Random_Bayes_Learn(len(train_data[0]), num_processes=8, dropout=1e-8)
b.learn_parameters(train_data, num_iterations=10)
true_results = a.test_network_fully_observed(test_data)
pred_results = b.test_network_fully_observed(test_data)
print(log_difference(true_results, pred_results))
# for i in range(1,5):
#     log_difference_set = []
#     file_path = dataset_path
#     for i in range(5):
#         b = POD_EM_Learn(dataset_path + '2', num_processes=8, dropout=None, stabilizer=1e-5)
#         train_data = Data_Extractor(dataset_path + 'train-p-' + str(i)).get_data()
#         b.learn_parameters(train_data)
#         pred_log = b.test_network_fully_observed(test_data)#
#         train_data = Data_Extractor(dataset_path + 'train-p-' + str(i)).get_data()
#         log_difference_set.append(log_difference(true_log, pred_log))
#     log_difference_list.append(log_difference_set)


# pretrained_likelihoods = b.test_network_fully_observed(test_data)
# trained_likelihoods = d.test_network_fully_observed(test_data)
# print(log_difference(pretrained_likelihoods, trained_likelihoods) / len(train_data))
