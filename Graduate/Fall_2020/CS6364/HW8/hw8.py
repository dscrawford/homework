# Made by Daniel Crawford
# Student Net ID: dsc160130
# Course: CS6364 - Artificial Intelligence

import torch
import torch.nn as nn
import numpy as np

# HYPERPARAMETERS
DISCOUNT_FACTOR = 0.7
NEGATIVE_REWARD = -5
INITIAL_STATE_PROB = 1 / 25

grid = np.array([
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1]
])


class GridGame2D:
    def __init__(self, grid):
        self.grid = grid
        self.x, self.y = grid.shape
        self.num_actions = 4
        self.terminal_states = {tuple(x) for x in np.argwhere(grid == 1)}

        self.actions = {
            0: lambda x, y: (x + 1, y) if x < self.x - 1 else (x, y),
            1: lambda x, y: (x - 1, y) if x > 0 else (x, y),
            2: lambda x, y: (x, y - 1) if y > 0 else (x, y),
            3: lambda x, y: (x, y + 1) if y < self.y - 1 else (x, y)
        }

    def action(self, x, y, a):
        if a > self.num_actions:
            raise Exception('Invalid Action')

        return self.actions[a](x, y)

    def get_index(self, x, y):
        return self.x * (game.x * self.y)

    def get_action_str(self, action):
        if action == 0:
            return 'right'
        elif action == 1:
            return 'left'
        elif action == 2:
            return 'up'
        else:
            return 'down'


class GridLearner2D:
    def __init__(self, gamma, r):
        self.gamma = gamma
        self.r = r
        self.Q = None

    def createQ(self, game):
        Q = np.zeros((game.x * game.y, 4))
        Q[:] = -5
        for state in game.terminal_states:
            index = state[1] + (state[0] * game.x)
            Q[index] = NEGATIVE_REWARD
            for a in range(game.num_actions):
                if game.action(state[0], state[1], a) == state:
                    Q[index][a] = 0
        return Q

    def train(self, game):
        Q = self.createQ(game)
        while True:
            newQ = np.zeros(Q.shape)
            for y in range(game.y):
                for x in range(game.x):
                    if (x, y) not in game.terminal_states:
                        for a in range(game.num_actions):
                            newX, newY = game.action(x, y, a)
                            newQ[x + (y * game.x)][a] = self.r + self.gamma * np.max(Q[newX + (newY * game.x)])
                    else:
                        newQ[x + (y * game.x)] = Q[x + (y * game.x)]
            if np.all(newQ == Q):
                break
            Q = newQ
        self.Q = Q

    def get_sequence(self, game, x, y, str='', action_num=1):
        num_turns = 0

        while (x, y) not in game.terminal_states:
            index = x + (y * game.x)
            actions = np.ndarray.flatten(np.argwhere(self.Q[index] == np.max(self.Q[index])))
            for action in range(1, len(actions) - 1):
                subX, subY = game.action(x, y, action)
                new_str = game.get_action_str(action) + ' --> '
                self.get_sequence(game, subX, subY, new_str, action_num)
                action_num += 1
            action = actions[0]
            str += game.get_action_str(action) + ' --> '
            x, y = game.action(x, y, action)
            num_turns += 1
            if num_turns > 100:
                break
        str += 'END'
        print('action ', action_num, ':', str)



game = GridGame2D(grid)
grid_learner = GridLearner2D(DISCOUNT_FACTOR, NEGATIVE_REWARD)

grid_learner.train(game)

grid_learner.get_sequence(game, 2, 2)
