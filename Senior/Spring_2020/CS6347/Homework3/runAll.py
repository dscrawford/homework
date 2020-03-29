import argparse
import varElim
import random
import time
# parser = argparse.ArgumentParser(description='Variable Elimination with sampling')
# parser.add_argument('filename', metavar='filename', type=str, help='A filename with its path(no extensions)')
# parser.add_argument('w', metavar='w', type=int, help='W-cutset value')
# parser.add_argument('N', metavar='N', type=int, help='Number of samples to take')
# parser.add_argument('--random_seed', metavar='random', type=int, help='Random seed, like it says')
# parser.add_argument('--adaptive', metavar='adaptive', type=bool,
#                     help='Whether to use adaptive distribution or not')
# args = parser.parse_args()
# fileName = args.filename
# w = args.w
# N = args.N
# random_seed = args.random_seed
# adaptive = args.adaptive

random.seed(10)
files = ['Grids_14', 'Grids_15', 'Grids_16', 'Grids_17', 'Grids_18']
N = [100, 1000, 10000, 20000]
M = [[0 for j in range(len(N) * 2)] for i in range(len(files))]
T = [[0 for j in range(len(N) * 2)] for i in range(len(files))]
for i, f in enumerate(files):
    print('FILE:', f, ':')
    network = varElim.GraphicalModel(f)
    for a in [0, 1]:
        for j, n in enumerate(N):
            print(n, a)
            t = time.time()
            M[i][j + a * len(N)] = network.sampleSumOut(i+1, n, a)
            T[i][j + a * len(N)] = time.time() - t
            print('time:', T[i][j], ', result: ', M[i][j])
print(M)
print(T)
