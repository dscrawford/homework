# Preparation

Please have python 3 and numpy installed.

One quirk, uai files are treated with expectation they always have .uai extension.
If you have a file 'filename.uai' then the input-uai-file will be filename not filename.uai

# Execution

python learning.py <input-uai-file> <task-id> <training-data> <test-data> [k - number of DAGs] [processes - number of processes] [no_verbose - turn off verbosity]

Example assuming you extract hw5-data.zip:

python learning.py hw5-data/dataset1/1 3 hw5-data/dataset1/train-f-4.txt hw5-data/dataset1/test.txt --k=6 --processes=8 --no_verbose

You may also use python learning.py -h to see list of commands.
