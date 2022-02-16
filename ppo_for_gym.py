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

mode = 'test'
modelsize = 'big'
games = ['CartPole-v1',
         'Pendulum-v0',
         'Acrobot-v1',
         'LunarLander-v2',
         'BipedalWalker-v3',
         'BipedalWalkerHardcore-v3',
         'hovering_control',
         'no_collision',]
game = games[7]
if game == games[6] or game == games[7]:
    env = rlschool.make_env('Quadrotor', task=game)
else:
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
std = 0.5
if mode == 'test': std = 0.2
act_size = 7.5
act_bias = 7.5


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def policy(self, x):
        if is_continue:
            x = torch.tanh(self.actor(x)) * act_size + act_bias
        else:
            x = F.softmax(self.actor(x), dim=-1)
        return x

    def evaluate(self, x):
        x = self.critic(x)
        return x

class NetPluc(nn.Module):
    def __init__(self):
        super(NetPluc, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def policy(self, x):
        if is_continue:
            x = torch.tanh(self.actor(x)) * act_size + act_bias
        else:
            x = F.softmax(self.actor(x), dim=-1)
        return x

    def evaluate(self, x):
        x = self.critic(x)
        return x

class Nethovering(nn.Module):
    def __init__(self):
        super(Nethovering, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.actormain = nn.Linear(64, 1)
        self.actoraid = nn.Linear(64, act_dim)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def policy(self, x):
        x = self.actor(x)
        main = self.actormain(x)
        main = torch.tanh(main)
        aid = torch.tanh(self.actoraid(x))
        return main, aid

    def evaluate(self, x):
        x = self.critic(x)
        return x

class Agent(object):
    def __init__(self):
        if modelsize == 'small':
            self.net = Net()
            self.old_net = Net()
        elif modelsize == 'big':
            self.net = NetPluc()
            self.old_net = NetPluc()
        self.old_net.load_state_dict(self.net.state_dict())
        self.data = []
        self.optimizer = torch.optim.Adam([
            {'params': self.net.actor.parameters(), 'lr': lr_actor},
            {'params': self.net.critic.parameters(), 'lr': lr_critic},
        ])
        self.std_arr = torch.eye(act_dim) * std * std
        std_arrs = []
        for i in range(horizon):
            std_arrs.append(self.std_arr.reshape(1, act_dim, act_dim))
        self.std_arrs = torch.cat(std_arrs, dim=0)

    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float)
        with torch.no_grad():
            if is_continue:
                mu = self.old_net.policy(s)
                dis = torch.distributions.MultivariateNormal(mu, self.std_arr)
                a = torch.clamp(dis.sample(), -act_size + act_bias, act_size + act_bias)
                log = dis.log_prob(a)
                return a.numpy(), log
            else:
                prob = self.old_net.policy(s)
                dis = torch.distributions.Categorical(prob)
                a = dis.sample()
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

            if is_continue:
                mu = self.net.policy(s)
                dis = torch.distributions.MultivariateNormal(mu, self.std_arrs)
                new_log = dis.log_prob(a).reshape(-1, 1)
                dis_entropy = dis.entropy()
            else:
                prob = self.net.policy(s)
                dis = torch.distributions.Categorical(prob)
                new_log = dis.log_prob(a.reshape(-1)).reshape(-1, 1)
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

class Agent_for_hovering(object):
    def __init__(self):
        self.net = Nethovering()
        self.old_net = Nethovering()
        self.old_net.load_state_dict(self.net.state_dict())
        self.data = []
        self.optimizer = torch.optim.Adam([
            {'params': self.net.actor.parameters(), 'lr': lr_actor},
            {'params': self.net.actoraid.parameters(), 'lr': lr_actor},
            {'params': self.net.actormain.parameters(), 'lr': lr_actor},
            {'params': self.net.critic.parameters(), 'lr': lr_critic},
        ])
        self.std_arr = torch.eye(act_dim) * std * std
        std_arrs = []
        for i in range(horizon):
            std_arrs.append(self.std_arr.reshape(1, act_dim, act_dim))
        self.std_arrs = torch.cat(std_arrs, dim=0)

        self.std_arr_for_main = torch.eye(1) * std * std
        std_arrs_for_main = []
        for i in range(horizon):
            std_arrs_for_main.append(self.std_arr_for_main.reshape(1, 1, 1))
        self.std_arrs_for_main = torch.cat(std_arrs_for_main, dim=0)

    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float)
        with torch.no_grad():
            main, aid = self.old_net.policy(s)
            main_dis = torch.distributions.MultivariateNormal(main, self.std_arr_for_main)
            a_main = main_dis.sample()
            main_log = main_dis.log_prob(a_main)
            aid_dis = torch.distributions.MultivariateNormal(aid, self.std_arr)
            a_aid = aid_dis.sample()
            aid_log = aid_dis.log_prob(a_aid)
            a = torch.clamp((a_main + 0.1 * a_aid) * act_size + act_bias, -act_size + act_bias, act_size + act_bias)
            log = aid_log + main_log
            return a.numpy(), log , a_main, a_aid

    def push_data(self, transition):
        self.data.append(transition)

    def sample(self):
        l_s, l_a, l_r, l_done, l_log, l_a_1, l_a_2 = [], [], [], [], [], [], []
        for trainsition in self.data:
            s, a, r, done, log, a_1, a_2 = trainsition
            l_s.append(torch.tensor([s], dtype=torch.float))
            if not is_continue: l_a.append(torch.tensor(a, dtype=torch.long).reshape(-1, 1))
            if is_continue: l_a.append(torch.tensor(a, dtype=torch.float).reshape(-1, act_dim))
            l_r.append(torch.tensor([[r]], dtype=torch.float))
            l_done.append(torch.tensor([[done]], dtype=torch.float))
            l_log.append(log.reshape(-1, 1))
            l_a_1.append(a_1.reshape(-1, 1))
            l_a_2.append(a_2.reshape(-1, 4))
        s, a, r, done, log, a_1, a_2 = map(torch.cat, [l_s, l_a, l_r, l_done, l_log, l_a_1, l_a_2])
        self.data = []
        return s, a, r, done, log, a_1, a_2

    def update(self):
        s, a, r, done, log, a_1, a_2 = self.sample()
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

            main, aid = self.net.policy(s)
            main_dis = torch.distributions.MultivariateNormal(main, self.std_arrs_for_main)
            main_dis_entropy = main_dis.entropy()
            main_log = main_dis.log_prob(a_1).reshape(-1, 1)
            aid_dis = torch.distributions.MultivariateNormal(aid, self.std_arrs)
            aid_dis_entropy = aid_dis.entropy()
            aid_log = aid_dis.log_prob(a_2).reshape(-1, 1)
            new_log = aid_log + main_log

            ratio = torch.exp(new_log - log)
            surr1 = ratio * A
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * A

            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(target, state_value) - 0.01 * main_dis_entropy - 0.01 * aid_dis_entropy

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
    if game == games[6] or game == games[7]:
        agent = Agent_for_hovering()
    else:
        agent = Agent()
    agent.load()
    for _ in range(1, MAX_EP + 1):
        s = env.reset()
        step = 0
        rewards = 0
        while step <= MAX_PER_EP:
            step += 1
            if game == games[6] or game == games[7]:
                a, log, a_1, a_2 = agent.choose_action(s)
                s_, r, done, info = env.step(a)
                agent.push_data((s, a, r, done, log, a_1, a_2))
            else:
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
    if game == games[6] or game == games[7]:
        agent = Agent_for_hovering()
    else:
        agent = Agent()
    agent.load()
    for _ in range(1, MAX_EP + 1):
        s = env.reset()
        step = 0
        rewards = 0
        while step <= MAX_PER_EP:
            step += 1
            env.render()
            if game == games[6] or game == games[7]:
                a, log, a_1, a_2 = agent.choose_action(s)
                s_, r, done, info = env.step(a)
            else:
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
