import torch
import torch.nn as nn
import torch.nn.functional as F


class DqnRelu(nn.Module):
    def __init__(self, state_size, action_size, seed=9090):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size
        dim_feed = 256
        dim_intermediate = 128
        dim_intermediate2 = 64
        dim_intermediate3 = 32
        # Layers
        self.feed_in = nn.Linear(self.state_size, dim_feed)
        self.feed_intermediate = nn.Linear(dim_feed, dim_intermediate)
        self.feed_intermediate2 = nn.Linear(dim_intermediate, dim_intermediate2)
        self.feed_intermediate3 = nn.Linear(dim_intermediate2, dim_intermediate3)
        self.out_act = nn.Linear(dim_intermediate3, self.action_size)

    def forward(self, x):
        dx = F.relu(self.feed_in(x))
        dx = F.relu(self.feed_intermediate(dx))
        dx = F.relu(self.feed_intermediate2(dx))
        dx = F.relu(self.feed_intermediate3(dx))
        dx = F.relu(self.out_act(dx))
        return dx


class DqnElu(nn.Module):
    def __init__(self, state_size, action_size, seed=9090):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size
        dim_feed = 256
        dim_intermediate = 128
        dim_intermediate2 = 64
        dim_intermediate3 = 32
        # Layers
        self.feed_in = nn.Linear(self.state_size, dim_feed)
        self.feed_intermediate = nn.Linear(dim_feed, dim_intermediate)
        self.feed_intermediate2 = nn.Linear(dim_intermediate, dim_intermediate2)
        self.feed_intermediate3 = nn.Linear(dim_intermediate2, dim_intermediate3)
        self.out_act = nn.Linear(dim_intermediate3, self.action_size)

    def forward(self, x):
        dx = F.elu(self.feed_in(x))
        dx = F.elu(self.feed_intermediate(dx))
        dx = F.elu(self.feed_intermediate2(dx))
        dx = F.elu(self.feed_intermediate3(dx))
        dx = F.elu(self.out_act(dx))
        return dx