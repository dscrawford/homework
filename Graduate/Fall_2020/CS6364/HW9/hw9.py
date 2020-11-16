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
LOSS_NEGATIVE_REWARD = -100
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
        self.N = grid_x * grid_y

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

    def get_one_hot_pos(self, y_pos, x_pos):
        return np.array([int((y, x) == (y_pos - 1, x_pos - 1)) for y in range(self.grid_y) for x in range(self.grid_x)])

    def set_pos(self, y, x):
        self.current_x = x
        self.current_y = y

    def get_action_str(self, action):
        return self.action_str[action]

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size, 8, bias=False),
            nn.ReLU(),
            nn.Linear(8, output_size, bias=False)
        ])



    def forward(self, x):
        y = x

        for layer in self.layers:
            y = layer(y)

        return y


game = CliffGame2D(6, 10, 6, 1, [(6, 10)], [(6, i) for i in range(2, 10)], NEGATIVE_REWARD, LOSS_NEGATIVE_REWARD)

Transition = namedtuple("Transition", ['y', 'x', 'action', 'reward', 'newY', 'newX', 'done'])

def init_weights_0(m):
    if isinstance(m, nn.Linear):
        # m.weight.data[:] = 0
        if m.bias is not None:
            m.bias.data[:] = 0

def reinforce(game, policy_model, target_model, target_criterion, policy_optimizer, target_optimizer,
              device, num_episodes=100, gamma=DISCOUNT_FACTOR, seq_limit=float('inf')):
    policy_model.apply(init_weights_0)
    target_model.apply(init_weights_0)
    input_size = game.N
    for episode in range(num_episodes):
        game.reset()
        # init episode
        episode_info = []

        y, x = game.get_pos()
        policy_model.eval()
        target_model.eval()
        with torch.no_grad():
            for t in count():
                state = torch.Tensor(game.get_one_hot_pos(y, x)).view(1, input_size).float().to(device)

                output = F.softmax(policy_model(state), dim=1).squeeze().cpu().numpy()

                # Choose best action
                action = np.random.choice(np.arange(len(output)), p=output)

                # Sample next state and check if finished
                newY, newX, reward, done = game.action(action)

                # Save episode information
                episode_info.append(Transition(y, x, action, reward, newY, newX, done))

                if done or t > seq_limit:
                    break

                x, y = newX, newY

        episode_reward = sum([t.reward for t in episode_info])
        episode_length = len(episode_info)
        print('EPISODE ', episode + 1, ' reward: ', episode_reward, ', length: ', episode_length, sep='')
        policy_model.train()
        target_model.train()

        states = [game.get_one_hot_pos(transition.y, transition.x) for transition in episode_info]
        actions = [transition.action for transition in episode_info]
        reward_returns = [sum(gamma**i * s.reward for i, s in enumerate(episode_info[t:])) for t in range(len(episode_info))]
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).view(len(episode_info), 1).to(device)
        reward_returns = torch.FloatTensor(reward_returns).view(len(episode_info), 1).to(device)
        baseline = target_model(states)
        advantage = reward_returns.clone() - baseline
        likelihoods = F.softmax(policy_model(states), dim=1)
        selected_likelihoods = likelihoods.gather(1, actions)

        policy_optimizer.zero_grad()
        policy_loss = -torch.log(selected_likelihoods) * advantage
        policy_loss.sum().backward(retain_graph=True)
        policy_optimizer.step()

        target_optimizer.zero_grad()
        target_loss = target_criterion(baseline, reward_returns)
        target_loss.backward()
        target_optimizer.step()

        # get_sequence(game, policy_model, device)

def actor_critic(game, policy_model, target_model, target_criterion, policy_optimizer, target_optimizer,
                 trace_decay_theta, trace_decay_w, device, num_episodes=100, gamma=DISCOUNT_FACTOR,
                 seq_limit=float('inf')):
    policy_model.apply(init_weights_0)
    target_model.apply(init_weights_0)
    input_size = game.N
    for episode in range(num_episodes):
        game.reset()
        # init episode
        episode_info = []

        y, x = game.get_pos()
        policy_model.train()
        target_model.train()

        z_theta = torch.Tensor([0 for _ in range(game.num_actions)]).to(device)
        z_w = torch.Tensor([0 for _ in range(1)]).to(device)
        I = torch.FloatTensor([1]).to(device)
        for t in count():
            state = torch.Tensor(game.get_one_hot_pos(y, x)).view(1, input_size).float().to(device)

            output = F.softmax(policy_model(state), dim=1)
            action = np.random.choice(np.arange(len(output.squeeze())), p=output.squeeze().detach().cpu().numpy())

            newY, newX, reward, done = game.action(action)

            new_state = torch.Tensor(game.get_one_hot_pos(newY, newX)).view(1, input_size).float().to(device)

            z_w = gamma * trace_decay_w * z_w + target_model(state)

            target_optimizer.zero_grad()
            z_w.squeeze().backward(retain_graph=True)
            target_optimizer.step()

            advantage = reward + gamma * target_model(new_state) - target_model(state)

            z_theta = gamma * trace_decay_theta * z_theta + I * torch.log(F.softmax(policy_model(state), dim=1))

            policy_optimizer.zero_grad()
            z_theta.sum().backward()
            policy_optimizer.step()

            I = gamma * I

            y, x = newY, newX


def get_sequence(game, policy_model, device):
    with torch.no_grad():
        for y in range(game.grid_y):
            for x in range(game.grid_x):
                state = torch.Tensor(game.get_one_hot_pos(y+1, x+1)).view(1, game.N).float().to(device)
                output = policy_model(state).squeeze().cpu().numpy()
                actions = np.ndarray.flatten(output == np.max(output))
                action_str = ''
                if actions[0]:
                    action_str += '→'
                if actions[1]:
                    action_str += '←'
                if actions[2]:
                    action_str += '↑'
                if actions[3]:
                    action_str += '↓'
                print(action_str, end='\t')
            print()


def q1(game):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_model = Model(game.N, game.num_actions).to(device)
    target_model = Model(game.N, 1).to(device)
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=0.001)
    target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.1)
    target_criterion = nn.SmoothL1Loss()

    reinforce(game, policy_model, target_model, target_criterion, policy_optimizer, target_optimizer,
              device, num_episodes=50, gamma=DISCOUNT_FACTOR)

    print('Final Result: ')
    get_sequence(game, policy_model, device)

def q2(game):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_model = Model(game.N, game.num_actions).to(device)
    target_model = Model(game.N, 1).to(device)
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=0.001)
    target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.1)
    target_criterion = nn.SmoothL1Loss()

    actor_critic(game, policy_model, target_model, target_criterion, policy_optimizer, target_optimizer, 0.9, 0.9,
              device, num_episodes=50, gamma=DISCOUNT_FACTOR)

    print('Final Result: ')
    get_sequence(game, policy_model, device)



q1(game)

q2(game)