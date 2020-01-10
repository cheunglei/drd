import math
import random
from collections import deque
import gym ,time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action

# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000,dt=1e-2):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.dt = dt
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

    def get_eval_action(self,action):
        return np.clip(action , self.low, self.high)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, 200)
        self.rnn = nn.LSTM(200, 300, 1, batch_first=True)
        self.linear2 = nn.Linear(300 + num_actions, 400)
        self.linear3 = nn.Linear(400, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        ob = self.linear1(state[:, 0, :])
        ob = ob.unsqueeze(1)
        for i in range(state.shape[1] - 1):
            ob_ = self.linear1(state[:, i + 1, :])
            ob_ = ob_.unsqueeze(1)
            ob = torch.cat([ob, ob_], 1)
        output, (h_n, c_n) = self.rnn(ob)
        output_in_last_timestep = h_n[-1, :, :]
        x = F.relu(output_in_last_timestep)
        x = torch.cat([x, action], 1)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs,200)
        self.rnn = nn.LSTM(200, 300 ,1 ,batch_first=True)
        self.linear2 = nn.Linear(300, num_actions)

        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        ob = self.linear1(state[:,0,:])
        ob = ob.unsqueeze(1)
        for i in range(state.shape[1]-1):
            ob_ = self.linear1(state[:,i+1,:])
            ob_ = ob_.unsqueeze(1)
            ob = torch.cat([ob, ob_], 1)
        output, (h_n, c_n) = self.rnn(ob)
        output_in_last_timestep = h_n[-1, :, :]
        x = F.relu(output_in_last_timestep)
        x = F.tanh(self.linear2(x))

        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)

        return action.detach().cpu().numpy()[0]

def update(batch_size,
                gamma=0.99,
                min_value=-np.inf,
                max_value=np.inf,
                soft_tau=1e-2,
                ):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(state).to(device))
    next_state = Variable(torch.FloatTensor(next_state).to(device))
    action = Variable(torch.FloatTensor(action).to(device))
    reward = Variable(torch.FloatTensor(reward).unsqueeze(1).to(device))
    done = Variable(torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device))

    policy_loss = value_net(state, policy_net(state))
    policy_loss = -policy_loss.mean()
    next_action = target_policy_net(next_state)
    target_value = target_value_net(next_state, next_action.detach())
    expected_value = reward + (1.0 - done) * gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = value_net(state, action)
    value_loss = value_criterion(value, expected_value.detach())

    policy_optimizer.zero_grad()
    policy_loss.backward()

    for param in policy_net.rnn.parameters():
        param.grad *= 1/math.sqrt(2)
    for param in value_net.rnn.parameters():
        param.grad *= 1/math.sqrt(2)

    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)
    nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=10)

    policy_optimizer.step()
    value_optimizer.zero_grad()
    value_loss.backward()

    for param in policy_net.rnn.parameters():
        param.grad *= 1 / math.sqrt(2)
    for param in value_net.rnn.parameters():
        param.grad *= 1/math.sqrt(2)
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)
    nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=10)

    value_optimizer.step()

    ##soft update
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


envid = "InvertedPendulum-v2"
state_length = 8  # Number of most recent frames to produce the input to the network
nb_epochs = 500
nb_epoch_cycles = 20
nb_rollout_steps = 100
nb_train_steps = 50
nb_eval = 5
seed = 0
value_lr = 1e-3
policy_lr = 1e-3
replay_buffer_size = 1000000
initial_size = 2000#10000
# max_frames  = 500
frame_idx = 0
batch_size = 64
eval_rewards = []
rewards_history = deque(maxlen=100)
epoch_episode_rewards = []
value_criterion = nn.MSELoss()
log = "_"+str(seed)
torch.manual_seed(seed)
random.seed(seed)

env = NormalizedActions(gym.make(envid))
ou_noise = OUNoise(env.action_space)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
env = NormalizedActions(gym.make(envid))
ou_noise = OUNoise(env.action_space)
eval_env = NormalizedActions(gym.make(envid))

#NET
value_net = ValueNetwork(state_dim, action_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, ).to(device)

#TARGET_NET
target_value_net = ValueNetwork(state_dim, action_dim).to(device)
target_policy_net = PolicyNetwork(state_dim, action_dim,).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)
value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
replay_buffer = ReplayBuffer(replay_buffer_size)

intial_state = env.reset()
zeroState = []
for _ in range(state_dim):
    zeroState.append(0)
states = [zeroState for _ in range(state_length-1)]
states.append(intial_state)
state = np.stack(states, axis=0)
#print(state.shape)
ou_noise.reset()

episode_reward = 0
start_time = time.time()

for epoch in range(nb_epochs):
    for cycle in range(nb_epoch_cycles):
        # Perform rollouts.
        for t_rollout in range(nb_rollout_steps):
            #env.render()
            action = policy_net.get_action(state)
            frame_idx += 1
            action = ou_noise.get_action(action, frame_idx)
            observation, reward, done, _ = env.step(action)
            observation_ = np.reshape(observation, (1, observation.shape[0]))
            next_state = np.append(state[1:, :], observation_, axis=0)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                epoch_episode_rewards.append(episode_reward)
                rewards_history.append(episode_reward)
                intial_state = env.reset()
                zeroState = []
                for _ in range(intial_state.shape[0]):
                    zeroState.append(0)
                states = [zeroState for _ in range(state_length - 1)]
                states.append(intial_state)
                state = np.stack(states, axis=0)
                episode_reward = 0

        # Train.
        for t_train in range(nb_train_steps):
            if len(replay_buffer) > initial_size:
                update(batch_size)


    ##eval
    eval_episode_reward_avg = 0
    for _ in range(nb_eval):
        eval_episode_reward = 0
        eval_intial_state = eval_env.reset()
        eval_zeroState = []
        for _ in range(state_dim):
            eval_zeroState.append(0)
        eval_states = [eval_zeroState for _ in range(state_length - 1)]
        eval_states.append(eval_intial_state)
        eval_state = np.stack(eval_states, axis=0)
        eval_done = False
        while not eval_done:
            #eval_env.render()
            eval_action = policy_net.get_action(eval_state)
            eval_action = ou_noise.get_eval_action(eval_action)
            eval_observation, eval_reward, eval_done, _ = eval_env.step(eval_action)
            eval_observation_ = np.reshape(eval_observation, (1, eval_observation.shape[0]))
            eval_next_state = np.append(eval_state[1:, :], eval_observation_, axis=0)
            eval_state = eval_next_state
            eval_episode_reward += eval_reward
        eval_episode_reward_avg += eval_episode_reward/nb_eval
    eval_rewards.append(eval_episode_reward_avg)

    print( " frames: " + str(frame_idx) + "  mean_reward:  " + str(np.mean(rewards_history))+"  eval_reward:  "+ str(eval_rewards[-1])+"  time:  " + str(time.time() - start_time))
    log_file = open(envid+log, "a+")
    log_file.write(" frames: " + str(frame_idx) + "  mean_reward:  " + str(np.mean(rewards_history))+"  eval_reward:  "+ str(eval_rewards[-1])+"  time:  " + str(time.time() - start_time)+"\n")
    log_file.close()
    start_time = time.time()
