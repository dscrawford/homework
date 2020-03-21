# SETUP

Please have the aclImdb folder or the train/test directory prepared, with folders for labels 'pos' and 'neg'
This project uses python3, sklearn, and argparser

# EXECUTION

python CS6320_Homework2.py <training directory> <test directory> <representation> <classifier> <stop words> [--regularization -r]

Regularization is an optional argument, please use for example '-r l1' at the end of the string to use it.

Example:
python CS6320_Homework2.py aclImdb/train aclImdb/test bow regression 1 -r l2

# NOTE

A python script 'runAll.py' as been included which will execute every possible combination for evaluation purposes. This requires you to have the aclImdb folder in the same directory as CS6320_Homework2.py.

After aclImdb is placed in there, simply run 'python runAll.py' to get all results back.
