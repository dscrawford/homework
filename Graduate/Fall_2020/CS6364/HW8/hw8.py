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
        return x * (self.x * y)

    def get_action_str(self, action):
        if action == 0:
            return 'right'
        elif action == 1:
            return 'left'
        elif action == 2:
            return 'up'
        else:
            return 'down'


action_num = 1


class GridLearner2D:
    def __init__(self, gamma, r):
        self.gamma = gamma
        self.r = r
        self.Q = None

    def train(self, game):
        Q = np.zeros((game.x * game.y, 4))
        while True:
            newQ = np.zeros(Q.shape)
            for y in range(game.y):
                for x in range(game.x):
                    for a in range(game.num_actions):
                        newX, newY = game.action(x, y, a)
                        if (newX, newY) in game.terminal_states:
                            newQ[x + (y * game.x)][a] = self.r
                        else:
                            newQ[x + (y * game.x)][a] = self.r + self.gamma * np.max(Q[newX + (newY * game.x)])
            if np.all(newQ == Q):
                break
            Q = newQ
        self.Q = Q

    def get_sequence(self, game, Q, x, y):
        action_num = {'value': 1}

        def get_sequence(game, Q, x, y, str=''):
            if (x, y) in game.terminal_states:
                str += 'END'
                print('action ', action_num['value'], ':', str)
                action_num['value'] += 1
            else:
                index = x + (y * game.x)
                actions = np.ndarray.flatten(np.argwhere(Q[index] == np.max(Q[index])))
                for i, action in enumerate(actions):
                    subX, subY = game.action(x, y, action)
                    new_str = str + game.get_action_str(action) + ' --> '
                    get_sequence(game, Q, subX, subY, new_str)

        get_sequence(game, Q, x, y)


class DeepGridLearner2D(nn.Module):
    def __init__(self, device, num_actions):
        super(DeepGridLearner2D, self).__init__()
        self.device = device
        self.layers = nn.ModuleList([
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, num_actions)
        ])

    def forward(self, x):
        y = torch.from_numpy(np.array([x])).float().to(self.device)

        for layer in self.layers:
            y = layer(y)

        return y


def q1():
    game = GridGame2D(grid)
    grid_learner = GridLearner2D(DISCOUNT_FACTOR, NEGATIVE_REWARD)
    grid_learner.train(game)
    grid_learner.get_sequence(game, grid_learner.Q, 2, 2)


def q2():
    game = GridGame2D(grid)
    num_epochs = 1000
    gamma = DISCOUNT_FACTOR
    r = NEGATIVE_REWARD
    eps = 1e-5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepGridLearner2D(device, game.num_actions).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    s = [np.random.randint(0, game.x), np.random.randint(0, game.y)]

    model.train()
    prev_loss = float('inf')
    for epoch in range(num_epochs):
        outputs = model(s)
        q_0, a = torch.max(outputs, 1)
        s = game.action(s[0], s[1], int(a[0].cpu()))
        q_1, _ = torch.max(model(s).data, 1)
        if s in game.terminal_states:
            y_i = torch.Tensor([r]).to(device)
            s = [np.random.randint(0, game.x), np.random.randint(0, game.y)]
        else:
            y_i = r + gamma * q_1

        print(q_0, q_1)
        optimizer.zero_grad()
        loss = criterion(q_0, y_i)
        loss.backward()
        optimizer.step()
        # if prev_loss - loss.item() < eps:
        #     break
        # else:
        #     prev_loss = loss.item()

        print(loss.item())
        print(s)

q2()
