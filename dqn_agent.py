import random
import torch
import torch.optim as O
from collections import deque, namedtuple
import torch.nn.functional as F

import numpy as np


from models import DqnElu, DqnRelu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExperienceBuffer(object):

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience_tuple = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = seed
        self.buffer = deque(maxlen=buffer_size)

    def append(self, state, action, reward, next_state, done):
        e = self.experience_tuple(state, action, reward, next_state, done)
        self.buffer.append(e)

    def getlast(self):
        current = random.sample(self.buffer, k=self.batch_size)
        s, a, r, sn, d = torch.from_numpy(np.vstack([e.state for e in current if e is not None])).float().to(device),\
             torch.from_numpy(np.vstack([e.action for e in current if e is not None])).long().to(device),\
             torch.from_numpy(np.vstack([e.reward for e in current if e is not None])).float().to(device),\
             torch.from_numpy(np.vstack([e.next_state for e in current if e is not None])).float().to(device), \
             torch.from_numpy(np.vstack([e.done for e in current if e is not None]).astype(np.uint8)).float().to(device)
        return s, a, r, sn, d

    def __len__(self):
        return len(self.buffer)


class DqnAgent(object):

    def __init__(self, state_size, action_size, seed=9090, gamma=0.99, batch_size=64, buffer_size=int(1e5),
                 update_rate=4, tau=1e-3, alpha=5e-4, mode=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_rate = update_rate
        self.tau = tau
        self.alpha = alpha
        # Experience buffer
        self.exp_buf = ExperienceBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
        # Network
        if mode == 0:
            self.q_local = DqnRelu(state_size, action_size, seed).to(device)
            self.q_target = DqnRelu(state_size, action_size, seed).to(device)
        else:
            self.q_local = DqnElu(state_size, action_size, seed).to(device)
            self.q_target = DqnElu(state_size, action_size, seed).to(device)
        # Network
        self.optimizer = O.Adam(self.q_local.parameters(), lr=alpha)
        # Time step
        self.t = 0

    def step(self, s, a, r, sn, done):
        # Add to experience buffer
        self.exp_buf.append(s, a, r, sn, done)
        self.t = (self.t + 1) % self.update_rate
        if self.t == 0:
            if len(self.exp_buf) > self.batch_size:
                batch = self.exp_buf.getlast()
                self.learn(batch)

    def learn(self, batch):
        s, a, r, sn, done = batch
        # Calculate Q in target and approximate networks
        Q_t_next = self.q_target(sn).detach().max(1)[0].unsqueeze(1)
        Q_t = r + (self.gamma * Q_t_next * (1 - done))
        Q_hat = self.q_local(s).gather(1, a)
        # Train
        loss = F.mse_loss(Q_hat, Q_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update target
        self.fixed_target_update(self.q_local, self.q_target)

    def act(self, state, epsilon=0.):
        s_ = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_local.eval()
        with torch.no_grad():
            a = self.q_local(s_)
        self.q_local.train()
        if random.random() > epsilon:
            return np.argmax(a.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def fixed_target_update(self, local, target):
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau)*target_param.data)

