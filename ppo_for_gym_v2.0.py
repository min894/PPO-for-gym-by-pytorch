# coding:utf-8

"""
    daixiangyu
    18357687883@163.com
"""

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import rlschool
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

mode = 'train'
modelsize = 'small'
games = ['CartPole-v1',
         'Pendulum-v0',
         'Acrobot-v1',
         'LunarLander-v2',
         'BipedalWalker-v3',
         'BipedalWalkerHardcore-v3',]
game = games[4]
env = gym.make(game)
is_continue = True
path = 'models/' + game + modelsize + '.pth'
if not is_continue: act_dim = env.action_space.n
if is_continue: act_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]
'''超参'''
MAX_EP = 100000
MAX_PER_EP = 1000
lr_actor = 0.0003
lr_critic = 0.001
horizon = 4000
K_epochs = 80
clip = 0.2
gamma = 0.99
if mode == 'test': std = 0.2
act_size = 1.
act_bias = 0



class NetaddStd(nn.Module):
    def __init__(self):
        super(NetaddStd, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mu = nn.Linear(64, act_dim)
        self.std = nn.Linear(64, 1)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def policy(self, x):
        x = self.actor(x)
        mu = torch.tanh(self.mu(x))
        std = torch.sigmoid(self.std(x))
        return mu, std

    def evaluate(self, x):
        x = self.critic(x)
        return x




class AgentaddStd(object):
    def __init__(self):
        self.net = NetaddStd()
        self.old_net = NetaddStd()
        self.old_net.load_state_dict(self.net.state_dict())
        self.data = []
        self.optimizer = torch.optim.Adam([
            {'params': self.net.actor.parameters(), 'lr': lr_actor},
            {'params': self.net.mu.parameters(), 'lr': lr_actor},
            {'params': self.net.std.parameters(), 'lr': lr_actor},
            {'params': self.net.critic.parameters(), 'lr': lr_critic},
        ])

    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float)
        with torch.no_grad():
            mu, std = self.old_net.policy(s)
            dis = torch.distributions.MultivariateNormal(mu, torch.eye(act_dim) * std * std)
            a = torch.clamp(dis.sample(), -act_size + act_bias, act_size + act_bias)
            log = dis.log_prob(a)
            return a.numpy(), log

    def push_data(self, transition):
        self.data.append(transition)

    def sample(self):
        l_s, l_a, l_r, l_done, l_log = [], [], [], [], []
        for trainsition in self.data:
            s, a, r, done, log = trainsition
            l_s.append(torch.tensor([s], dtype=torch.float))
            if not is_continue: l_a.append(torch.tensor(a, dtype=torch.long).reshape(-1, 1))
            if is_continue: l_a.append(torch.tensor(a, dtype=torch.float).reshape(-1, act_dim))
            l_r.append(torch.tensor([[r]], dtype=torch.float))
            l_done.append(torch.tensor([[done]], dtype=torch.float))
            l_log.append(log.reshape(-1, 1))
        s, a, r, done, log = map(torch.cat, [l_s, l_a, l_r, l_done, l_log])
        self.data = []
        return s, a, r, done, log

    def update(self):
        s, a, r, done, log = self.sample()
        with torch.no_grad():
            target = []
            tmp = 0.
            r = r.numpy()
            done = done.numpy()
            for r_, done_ in zip(r[::-1], done[::-1]):
                tmp = tmp * gamma * (1 - done_) + r_
                target.insert(0, tmp)
            target = torch.tensor(target).reshape(-1, 1)
            target = (target - target.mean()) / (target.std() + 1e-7)

        for i in range(K_epochs):
            state_value = self.net.evaluate(s)
            A = target - state_value.detach()

            mu, std = self.net.policy(s)
            std_arrs = []
            for i in std:
                std_arrs.append((i * i * torch.eye(act_dim).reshape(1, act_dim, act_dim)))
            std_arrs = torch.cat(std_arrs)
            dis = torch.distributions.MultivariateNormal(mu, std_arrs)
            new_log = dis.log_prob(a).reshape(-1, 1)
            dis_entropy = dis.entropy()

            ratio = torch.exp(new_log - log)

            surr1 = ratio * A
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * A

            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(target, state_value) - 0.01 * dis_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_net.load_state_dict(self.net.state_dict())

    def save(self):
        torch.save(self.net.state_dict(), path)

    def load(self):
        try:
            self.net.load_state_dict(torch.load(path))
            self.old_net.load_state_dict(torch.load(path))
            print('...load...')
        except:
            pass

    def __len__(self):
        return len(self.data)


def train():
    agent = AgentaddStd()
    agent.load()
    for _ in range(1, MAX_EP + 1):
        s = env.reset()
        step = 0
        rewards = 0
        while step <= MAX_PER_EP:
            step += 1
            a, log = agent.choose_action(s)
            s_, r, done, info = env.step(a)
            agent.push_data((s, a, r, done, log))
            rewards += r
            if agent.__len__() >= horizon:
                agent.update()

            if done:
                break
            s = s_
        if _ % 10 == 0:
            print('episode:', _, 'rewards:', rewards)
        if _ % 10 == 0:
            agent.save()


def test():
    agent = AgentaddStd()
    agent.load()
    for _ in range(1, MAX_EP + 1):
        s = env.reset()
        step = 0
        rewards = 0
        while step <= MAX_PER_EP:
            step += 1
            env.render()
            a, log = agent.choose_action(s)
            s_, r, done, info = env.step(a)
            rewards += r
            if done:
                break
            s = s_
        print('episode:', _, 'rewards:', rewards)


if __name__ == '__main__':
    if mode == 'train':
        train()
    elif mode == 'test':
        test()
