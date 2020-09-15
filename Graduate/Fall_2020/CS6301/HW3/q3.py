import matplotlib.pyplot as plt
import random
import numpy as np

TOTAL_QUESTIONS = 100
TOTAL_TRIALS = 1000
C = list(range(1, TOTAL_QUESTIONS + 1))

# mu = lambda C: (TOTAL_QUESTIONS - C) * 0.25
# std = lambda C: np.sqrt(TOTAL_QUESTIONS - C) * (3 / 16)
# true_mus, true_stds = np.transpose([(mu(c), std(c)) for c in C])
# plt.plot(C, true_mus)
# plt.plot(C, true_stds)


def simulate_n_times(c):
    questions_left = TOTAL_QUESTIONS - c
    scores = [sum([random.random() <= 0.25 for _ in range(questions_left)]) for _ in range(TOTAL_TRIALS)]
    mu = sum(scores) / TOTAL_TRIALS
    std = np.sqrt(sum((np.array(scores) - mu)**2) / TOTAL_TRIALS)
    return mu, std


sim_mus, sim_stds = np.transpose([simulate_n_times(c) for c in C])
plt.plot(C, sim_mus)
plt.plot(C, sim_stds)
plt.xlabel('average # of questions known')
plt.ylabel('average # of questions guessed right')
plt.show()
