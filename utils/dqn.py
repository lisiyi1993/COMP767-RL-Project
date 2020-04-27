#!/usr/bin/env python
# coding: utf-8

import os
import math
import time
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.autograd as autograd
from torch.distributions import Categorical

from minipacman import MiniPacman
from multiprocessing_env import SubprocVecEnv

import matplotlib.pyplot as plt
from IPython import display
from IPython.core.debugger import set_trace

from tqdm import tqdm


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))



def select_action(state, policy_net, num_actions, num_envs, epilson=0.9):
    e = random.random()
    
    if e < epilson:
        with torch.no_grad():
            return policy_net.act(state)
    else:
        # return torch.tensor([[random.randrange(num_actions)]], device=DEVICE, dtype=torch.long)
        return torch.from_numpy(np.random.randint(num_actions, size=(num_envs, 1))).long().to(DEVICE)


def optimize_model(policy_net, target_net, memory, batch_size=128, gamma=0.999):
    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                  device=DEVICE, dtype=torch.bool)
    
    non_final_next_states = torch.cat([torch.FloatTensor([s]).to(DEVICE) for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=DEVICE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    
    # Compute Huber loss
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # loss = F.mse_loss(state_action_values, expected_state_action_values.view(-1, 1))
    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    target_net.eval()
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step() 


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):
    def __init__(self, in_shape, n_actions):
        super(DQN, self).__init__()
        self.in_shape = in_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            # try adding batch norm
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
        )
        
        self.head = nn.Linear(256, n_actions)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        value = self.head(x)
        return value
    

    def act(self, x):
        value = self.forward(x)
        # probs = F.softmax(value, dim=-1)
        return value.max(1)[1].view(-1, 1)
        
        # return probs.multinomial(1)
    

    def feature_size(self):
        return self.features(torch.zeros(1, *self.in_shape)).view(1, -1).size(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=int(2e5), type=int)
    parser.add_argument("--update_freq", default=1000, type=int)
    args = parser.parse_args()
    PATH = os.path.join("..", "training", f"dqn_freq_{target_update}")

    mode = "regular"
    num_envs = 16
    def make_env():
        def _thunk():
            env = MiniPacman(mode, 1000)
            return env

        return _thunk


    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    state_shape = envs.observation_space.shape
    num_actions = envs.action_space.n


    policy_net = DQN(state_shape, num_actions).to(DEVICE) # save to checkpoint
    target_net = DQN(state_shape, num_actions).to(DEVICE) # save to checkpoint
    target_net.load_state_dict(policy_net.state_dict()) # save to checkpoint
    target_net.eval()
    policy_net.train()


    lr    = 7e-4
    eps   = 1e-5
    alpha = 0.99

    optimizer = optim.RMSprop(policy_net.parameters(), lr, eps=eps, alpha=alpha) # save to checkpoint
    memory = ReplayMemory(10000) # save to checkpoint


    num_frames = args.epoch
    target_update = args.update_freq
    batch_size = 512
    backprops_freq = 0

    
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    
    all_rewards = []

    episode_rewards = torch.zeros(num_envs, 1).to(DEVICE)
    final_rewards   = torch.zeros(num_envs, 1).to(DEVICE)

    state = envs.reset()
    state = torch.FloatTensor(np.float32(state)).to(DEVICE)

    for i_update in tqdm(range(num_frames+1), total=num_frames+1):
        for t in range(10):
            # Select and perform an action
            action = select_action(state, policy_net, num_actions, num_envs)
            next_state, reward, done, _ = envs.step(action.squeeze(1).cpu().data.numpy())
            reward = torch.tensor(reward).unsqueeze(1).to(DEVICE)
            
            episode_rewards += reward
            masks = torch.FloatTensor(1-np.array(done)).unsqueeze(1).to(DEVICE)
            final_rewards *= masks
            final_rewards += (1-masks) * episode_rewards
            episode_rewards *= masks
            
            # Store the all transition in memory
            for i in range(num_envs):
                memory.push(state[i].unsqueeze(0), action[i].unsqueeze(0), next_state[i], reward[i])
        
            # Move to the next state
            state = torch.FloatTensor(np.float32(next_state)).to(DEVICE)

            # Perform one step of the optimization (on the target network)
            optimize_model(policy_net, target_net, memory, batch_size=batch_size)
                 
            # Update the target network, copying all weights and biases in DQN
            backprops_freq += 1
            if backprops_freq % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        
        if i_update % 100 == 0:
            print(f"{i_update} Done!")

            all_rewards.append(final_rewards.mean())
            
            plt.figure(figsize=(6, 5))
            plt.plot(all_rewards)
            plt.title(f"Training Rewards (DQN)")
            plt.xlabel("Environmental Steps x1000")
            plt.ylabel("Rewards")
            plt.savefig(os.path.join(PATH, "training_reward.pdf"), bbox_inches="tight")
            plt.close()

            torch.save({
                        'policy_net': policy_net.state_dict(),
                        'target_net': target_net.state_dict(),
                        'RMSprop_state_dict': optimizer.state_dict(),
                        'all_rewards': all_rewards,
                        'episode_rewards': episode_rewards,
                        'final_rewards': final_rewards
                        }, os.path.join(PATH, "dqn_checkpoint"))



