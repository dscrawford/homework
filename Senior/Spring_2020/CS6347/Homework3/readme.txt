# SETUP

Please have UAI file format files in the same file as varElim.py.
Regex and numpy python library is needed for this project as well as python 3

EX:
 1.uai
 1.uai.evid

# Execution
python3 ./varElim.py <fileName without extensions> <wcutset value> <number of samples to take> [random_seed] [adaptive]
EX:
 python3 ./varElim.py 3 1 100 --random_seed 12345678 --adaptive True
