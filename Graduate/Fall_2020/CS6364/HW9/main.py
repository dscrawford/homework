# %%

# Made by Daniel Crawford
# Student Net ID: dsc160130
# Course: CS6364 - Artificial Intelligence

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

# hyperparameters
DISCOUNT_FACTOR = 0.7
NEGATIVE_REWARD = -5
INITIAL_STATE_PROB = 1 / 25

RIGHT = 0
LEFT  = 1
UP    = 2
DOWN  = 3

# Cliff Game 2D
# 2D grid with 3 different positions
# 0 - Empty space
# 1 - Target space
# 2 - Cliff space
class CliffGame2D:
    def __init__(self, grid_y, grid_x, start_y, start_x, target_pos_list, cliff_pos_list, base_reward, lose_reward):
        # Initialize grid space
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid = np.zeros((grid_y, grid_x), dtype=int)
        self.start_x, self.start_y = start_x - 1, start_y - 1
        self.current_x, self.current_y = self.start_x, self.start_y
        for y, x in target_pos_list:
            self.grid[y - 1][x - 1] = 1
        for y, x in cliff_pos_list:
            self.grid[y - 1][x - 1] = 2
        self.num_actions = 4
        self.base_reward = base_reward
        self.lose_reward = lose_reward

        # Create actions list which can either be up, down, left, or right
        self.actions = {
            RIGHT: lambda y, x: (y, x + 1) if x + 1 < grid_x else (y, x),
            LEFT: lambda y, x: (y, x - 1) if x - 1 >= 0 else (y, x),
            UP: lambda y, x: (y - 1, x) if y - 1 > 0 else (y, x),
            DOWN: lambda y, x: (y + 1, x) if y + 1 < grid_y else (y, x)
        }

    def action(self, action):
        # Perform action
        self.current_y, self.current_x = self.actions[action](self.current_y, self.current_x)

        # Compute rewards for new position
        done = self.grid[self.current_y][self.current_x] == 1 or self.grid[self.current_y][self.current_x] == 2

        reward = self.base_reward if self.grid[self.current_y][self.current_x] != 2 else self.lose_reward

        return self.current_y + 1, self.current_x + 1, reward, done

    def str(self):
        # Create string of current state of board
        # 0 - Free space
        # 1 - Terminal space
        # 2 - Cliff space
        # x - player space
        # d - player done
        # l - player lost
        grid_str = ''
        for y in range(self.grid_y):
            for x in range(self.grid_x):
                if (y, x) == (self.current_y, self.current_x):
                    letter = 'd' if self.grid[y][x] == 1 else ('l' if self.grid[y][x] == 2 else 0)
                    grid_str += letter + '\t'
                else:
                    grid_str += str(self.grid[y][x]) + '\t'
            grid_str += '\n'
        return grid_str

    def reset(self):
        self.current_x, self.current_y = self.start_x, self.start_y

    def get_pos(self):
        return self.current_y + 1, self.current_x + 1


game = CliffGame2D(6, 10, 6, 1, [(6, 10)], [(6, i) for i in range(2, 10)], NEGATIVE_REWARD, NEGATIVE_REWARD * 20)
print(game.get_pos())


def q1():
    pass