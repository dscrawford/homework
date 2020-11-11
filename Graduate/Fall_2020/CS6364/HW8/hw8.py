#%%

# Made by Daniel Crawford
# Student Net ID: dsc160130
# Course: CS6364 - Artificial Intelligence

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

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

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, game, x, y, action):
        newX, newY = game.action(x, y, action)
        if self.capacity < len(self.memory):
            self.memory[self.position] = [x, y, action, newX, newY]
            self.position = (self.position + 1) % self.capacity
        self.memory.append([x, y, action, newX, newY])

    def sample(self, size):
        return np.array(self.memory)[np.random.randint(0, len(self.memory), size)]


def generate_coordinates(game):
    x, y = np.random.randint(0, game.x), np.random.randint(0, game.y)
    while (x, y) in game.terminal_states:
        x, y = np.random.randint(0, game.x), np.random.randint(0, game.y)
    return x, y


def q1():
    game = GridGame2D(grid)
    grid_learner = GridLearner2D(DISCOUNT_FACTOR, NEGATIVE_REWARD)
    grid_learner.train(game)
    for y in range(game.y):
        for x in range(game.x):
            index = x + game.x * y
            max_points = np.ndarray.flatten(np.argwhere(grid_learner.Q[index] == np.max(grid_learner.Q[index])))
            action_str = ''
            for action in max_points:
                if action == 0:
                    action_str += '→'
                elif action == 1:
                    action_str += '←'
                elif action == 2:
                    action_str += '↑'
                elif action == 3:
                    action_str += '↓'
            print(action_str, end='\t')
        print('\n')


def q2():
    game = GridGame2D(grid)
    num_episodes = 1000
    gamma = DISCOUNT_FACTOR
    r = NEGATIVE_REWARD
    sequence_limit = 5
    target_update = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy_model = DeepGridLearner2D(game.num_actions).to(device)
    target_model = DeepGridLearner2D(game.num_actions).to(device)
    target_model.eval()
    target_model.load_state_dict(policy_model.state_dict())

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(policy_model.parameters(), lr=0.01)

    for episode in range(num_episodes):
        s = torch.Tensor([generate_coordinates(game)]).float().to(device)
        in_terminal = False
        sequence_i = 0

        while not in_terminal:
            q_0, action = policy_model(s).max(1)
            newX, newY = game.action(int(s[0][0].cpu()), int(s[0][1].cpu()), int(action.cpu()))
            in_terminal = (newX, newY) in game.terminal_states
            reward = torch.Tensor([r]).float().to(device)
            expected_value = reward
            if not in_terminal:
                expected_value += target_model(
                    torch.Tensor([[newX, newY]]).float().to(device)
                ).max().squeeze().to(device) * gamma

            loss = criterion(q_0, expected_value)
            optimizer.zero_grad()
            loss.backward()
            for param in policy_model.parameters():
                param.grad.data.clamp(-1, 1)
            optimizer.step()

            if (newX, newY) in game.terminal_states:
                in_terminal = True

            sequence_i += 1
            if sequence_i >= sequence_limit:
                x, y = generate_coordinates(game)
                s = torch.Tensor(
                    [[x, y]]
                ).float().to(device)
                sequence_i = 0
            else:
                s = torch.Tensor(
                    [[newX, newY]]
                ).float().to(device)

        if (episode + 1) % target_update == 0:
            target_model.load_state_dict(policy_model.state_dict())

    with torch.no_grad():
        for y in range(game.y):
            for x in range(game.x):
                output = policy_model(torch.from_numpy(np.array([[x, y]])).float().to(device)).cpu().numpy()
                max_points = np.ndarray.flatten(np.argwhere(output == np.max(output)))
                action_str = ''
                for action in max_points:
                    if action == 0:
                        action_str += '→'
                    elif action == 1:
                        action_str += '←'
                    elif action == 2:
                        action_str += '↑'
                    elif action == 3:
                        action_str += '↓'
                print(action_str, end='\t')
            print('\n')


def q3(num_episodes=100, sequence_limit=5, target_update=10, batch_size=32, eps=1e-2):
    game = GridGame2D(grid)
    gamma = DISCOUNT_FACTOR
    r = NEGATIVE_REWARD
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replay = ReplayMemory(1000)

    policy_model = DeepGridLearner2D(game.num_actions).to(device)
    target_model = DeepGridLearner2D(game.num_actions).to(device)
    target_model.eval()
    target_model.load_state_dict(policy_model.state_dict())

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(policy_model.parameters(), lr=0.01, momentum=0.9)

    tq = tqdm(range(num_episodes))

    for episode in tq:
        x, y = generate_coordinates(game)
        sequence_i = 0
        while (x, y) not in game.terminal_states:
            sequence_i += 1
            s = torch.Tensor([[x, y]]).float().to(device)
            with torch.no_grad():
                if np.random.uniform() > eps and sequence_i < sequence_limit:
                    _, action = policy_model(s).max(1)
                    action = int(action.cpu())
                else:
                    sequence_i = 0
                    action = np.random.randint(game.num_actions)
            newX, newY = game.action(x, y, action)

            replay.push(game, x, y, action)

            batch = replay.sample(batch_size)

            states = torch.Tensor(batch[:, 0:2]).float().to(device)
            actions = torch.Tensor(batch[:, 2:3]).view(batch_size, 1).type(torch.int64).to(device)
            new_states = torch.Tensor(batch[:, 3:5]).float().to(device)

            q_0 = policy_model(states).gather(1, actions)

            expected_value = torch.full((batch_size, ), r, dtype=torch.float64).to(device)
            non_terminal_mask = torch.Tensor(
                [(int(new_states[i][0].cpu()), int(new_states[i][1].cpu())) not in game.terminal_states for i in range(batch_size)]
            ).bool().to(device)
            expected_value[non_terminal_mask] += target_model(new_states[non_terminal_mask]).max(1)[0] * gamma

            optimizer.zero_grad()
            loss = criterion(q_0, expected_value.unsqueeze(1))
            loss.backward()
            optimizer.step()

            x, y = newX, newY
        if (episode + 1) % target_update == 0:
            target_model.load_state_dict(policy_model.state_dict())

    with torch.no_grad():
        for y in range(game.y):
            for x in range(game.x):
                output = policy_model(torch.from_numpy(np.array([[x, y]])).float().to(device)).cpu().numpy()
                output = np.ndarray.flatten(output)
                max_points = np.ndarray.flatten(np.argwhere(output == np.max(output)))
                action_str = ''
                for action in max_points:
                    if action == 0:
                        action_str += '→'
                    elif action == 1:
                        action_str += '←'
                    elif action == 2:
                        action_str += '↑'
                    elif action == 3:
                        action_str += '↓'
                print(action_str, end='\t')
            print('\n')

q3(200, 10, 10, 16, 1e-2)