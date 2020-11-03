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
    def __init__(self, num_actions):
        super(DeepGridLearner2D, self).__init__()
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
        y = x

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
    num_epochs = 200
    gamma = DISCOUNT_FACTOR
    r = NEGATIVE_REWARD
    sequence_threshold = 5
    BATCH_SIZE = 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepGridLearner2D(game.num_actions).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    s_0 = []
    for i in range(BATCH_SIZE):
        while True:
            x, y = np.random.randint(0, game.x), np.random.randint(0, game.y)
            if (x, y) not in game.terminal_states:
                break
        s_0.append([x,y])
        
    s = np.array(s_0)
    sequence_count = np.zeros(len(s))

    model.train()
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        outputs = model(torch.from_numpy(s).float().to(device))
        q_0, a = torch.max(outputs, 1)
        s_n = np.array([game.action(s[i][0], s[i][1], int(a[i].cpu())) for i in range(len(a))])

        y = []
        for i in range(len(s_n)):
            if tuple(s_n[i]) in game.terminal_states:
                y_i = r
                s_n[i] = (np.random.randint(0, game.x), np.random.randint(0, game.y))
                sequence_count[i] = 0
            else:
                q_1, _ = torch.max(model(torch.from_numpy(np.array([s_n[i]])).float().to(device)), 1)
                y_i = (r + gamma * float(q_1.cpu()))
            y.append(y_i)
        y = torch.Tensor(y).float().to(device)

        loss = criterion(q_0, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        sequence_count += 1
        for i in range(len(s_n)):
            if sequence_count[i] > sequence_threshold:
                s_n[i] = (np.random.randint(0, game.x), np.random.randint(0, game.y))
                sequence_count[i] = 0
        s = s_n

        print('EPOCH ', epoch + 1, ', Loss: ', loss.item())

    problems = [[0, 1], [1, 0], [4, 3]]
    for problem in problems:
        _, a = torch.max(model(torch.from_numpy(np.array([problem])).float().to(device)), 1)
        print(problem, 'action: ', game.get_action_str(int(a.cpu())))


q2()
