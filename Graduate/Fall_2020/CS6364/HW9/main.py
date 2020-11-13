# %%

# Made by Daniel Crawford
# Student Net ID: dsc160130
# Course: CS6364 - Artificial Intelligence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from itertools import count
from collections import namedtuple

# hyperparameters
DISCOUNT_FACTOR = 0.7
NEGATIVE_REWARD = -5
INITIAL_STATE_PROB = 1 / 25

RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3


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

        self.action_str = {
            RIGHT: 'right',
            LEFT: 'left',
            UP: 'up',
            DOWN: 'down'
        }

    def action(self, action):
        # Perform action
        self.current_y, self.current_x = self.actions[action](self.current_y, self.current_x)

        val = self.grid[self.current_y][self.current_x]
        # Compute rewards for new position
        done = val == 1
        #
        # if val == 2:
        #     self.reset()

        reward = self.base_reward if val != 2 else self.lose_reward

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

    def get_action_str(self, action):
        return self.action_str[action]



class Model(nn.Module):
    def __init__(self, output_size):
        super(Model, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_size)
        ])



    def forward(self, x):
        y = x

        for layer in self.layers:
            y = layer(y)

        return y


game = CliffGame2D(6, 10, 6, 1, [(6, 10)], [(6, i) for i in range(2, 10)], NEGATIVE_REWARD, -100)

Transition = namedtuple("Transition", ['y', 'x', 'action', 'reward', 'newY', 'newX', 'done'])


def reinforce(game, policy_model, target_model, policy_criterion, target_criterion, policy_optimizer, target_optimizer,
              device, num_episodes=100, gamma=DISCOUNT_FACTOR, seq_limit=None):
    seq_limit = float('inf') if seq_limit is None else seq_limit

    for episode in range(num_episodes):
        game.reset()
        # init episode
        episode_info = []

        y, x = game.get_pos()
        state = torch.Tensor([y, x]).view(1,2).float().to(device)
        # true_sequence = [UP, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, DOWN]
        true_sequence = LEFT
        total_reward = 0
        policy_model.eval()
        target_model.eval()
        with torch.no_grad():
            for t in count():
                # Choose best action
                output = F.softmax(policy_model(state), dim=1)[0].cpu().numpy()
                action = np.random.choice(np.arange(len(output)), p=output)
                # action = UP
                # Sample next state, reward and check if finished
                newY, newX, reward, done = game.action(action)

                # Save episode information
                episode_info.append(Transition(y, x, action, reward, newY, newX, done))

                total_reward += reward

                if done or t > seq_limit:
                    break
                x, y = newX, newY

        policy_model.train()
        target_model.train()
        policy_loss = torch.Tensor([0]).to(device)
        target_loss = torch.Tensor([0]).to(device)
        policy_optimizer.zero_grad()
        target_optimizer.zero_grad()
        for t, transition in enumerate(episode_info):
            state = torch.Tensor([transition.y, transition.x]).view(1,2).float().to(device)
            baseline = target_model(state).squeeze()
            pred_return = F.softmax(policy_model(state), dim=1).squeeze()

            # Get the expected value at this current state
            total_return = sum(gamma ** i * s.reward for i, s in enumerate(episode_info[t:]))

            # Get a baseline value for transition state
            advantage = total_return - float(baseline.cpu())

            # Update policy model
            policy_loss += -torch.log(pred_return.gather(0, torch.Tensor([transition.action]).type(torch.int64).to(device))) * total_return

            # Update target model
            target_loss += target_criterion(baseline, torch.Tensor([total_return]).squeeze().to(device))
        policy_loss /= len(episode_info)
        target_loss /= len(episode_info)
        print(policy_loss)
        policy_loss.backward()
        target_loss.backward()
        policy_optimizer.step()
        target_optimizer.step()
        print('EPISODE ', episode, ' reward: ', total_reward, sep='')

        print('START ', end='')
        for transition in episode_info:
            print('-> ', game.get_action_str(transition.action), ' ', end='')
        print()


def get_sequence(game, policy_model, device):
    game.reset()
    print('START ', end='')
    for t in count():
        y, x = game.get_pos()
        state = torch.Tensor([y, x]).view(1, 2).float().to(device)
        _, action = policy_model(state).max(1)
        action = int(action[0])
        _, _, _, done = game.action(action)
        print('->', game.get_action_str(action), ' ', end='')
        if t > 100 or done:
            break

def q1(game):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_model = Model(game.num_actions).to(device)
    target_model = Model(1).to(device)
    policy_optimizer = torch.optim.SGD(policy_model.parameters(), lr=0.01)
    target_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.01)
    policy_criterion = nn.CrossEntropyLoss()
    target_criterion = nn.SmoothL1Loss()

    reinforce(game, policy_model, target_model, policy_criterion, target_criterion, policy_optimizer, target_optimizer,
              device, num_episodes=100, gamma=DISCOUNT_FACTOR)

    get_sequence(game, policy_model, device)




q1(game)