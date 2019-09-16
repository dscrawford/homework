Options to run the code:

python Homework_1_Code <File Name> <Decision Tree> <Impurity Function> <Pruning Choices>
python Homework_1_Code <File Name> <Random Forest>

File Name: Substring with clauses and samples
EX: 'c300_d100'

Decision Tree: String for decision tree
EX: 'decisiontree' or 'dt'

Random Forest: String for random forest
EX: 'randomforest' or 'rf'

Impurity Function: Choice of impurity function entropy or variance impurity
EX for Entropy: 'e' or 'entropy'
EX for Variance impurity: 'vi' or 'varianceimpurity'

Pruning Choices: Comma separated choices of pruning algorithms(Can avoid re-creating model several times).
Options: 'reducederrorpruning' or 'rep' or 'reducederror', 'depthbasedpruning' or 'dbp' or 'depthbased', 'naive' or 'n'
EX: rep,dbp,n
