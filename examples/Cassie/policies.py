import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal

class VisionFrontend2D(nn.Module):
    def __init__(self, im_shape):
        super(VisionFrontend2D, self).__init__()
        ksize = 5
        in_height = im_shape[0]
        in_width = im_shape[1]
        self.conv_input = nn.Sequential(nn.Conv2d(1, 8, ksize),
                                nn.ReLU(),
                                nn.Conv2d(8, 16, ksize),
                                nn.ReLU(),
                                nn.Conv2d(16, 32, ksize),
                                nn.ReLU())

        test = self.conv_input(torch.zeros((1, 1, in_height, in_width)))
        # for now, just a single linear layer
        self.output_size = 128
        self.linear_layer = nn.Sequential(torch.nn.Linear(torch.numel(test), self.output_size))

    # x should be of shape(batch, 1, height, width)
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.conv_input(x)
        x = x.view(x.size(0), -1)
        return self.linear_layer(x)


class StateEncoder(nn.Module):
    def __init__(self, vision_net, state_dim, state_encoding_dim, output_dim):
        super(StateEncoder, self).__init__()
        self.vision_net = vision_net
        
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.state_encoding_dim = state_encoding_dim
        self.state_net = nn.Sequential(
                            nn.Linear(state_dim, 16),
                            nn.ReLU(),
                            nn.Linear(16, state_encoding_dim),
                            nn.ReLU())

        self.encoder_head = nn.Sequential(
                                nn.Linear(self.vision_net.output_size + self.state_encoding_dim, 64),
                                nn.ReLU(),
                                nn.Linear(64, output_dim),
                                nn.ReLU())


    # x = (state_batch, terrain_batch) batched
    def forward(self, x):
        x_state, x_terrain = x[0], x[1]
        if len(x_state.size()) == 1:
            x_state = x_state.view(1, x_state.size(0)) 
        s = self.state_net(x_state)
        t = self.vision_net(x_terrain)
        concat = torch.cat([s, t], dim = 1)
        return self.encoder_head(concat)


class Actor_TD3(nn.Module):
    def __init__(self, state_encoder, a_dim):
        super(Actor_TD3, self).__init__()
        self.encoder = state_encoder
        self.fc_mu = nn.Sequential(nn.Linear(self.encoder.output_dim, a_dim),
                                   nn.Sigmoid())


    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x) * 2 - 1 ## going from (0, 1) to (-1, 1)
        return mu


class QNet_TD3(nn.Module):
    def __init__(self, state_encoder, action_dim):
        super(QNet_TD3, self).__init__()
        # encoder fills in the role for fc_s
        self.encoder = state_encoder
        
        # Todo: make the encoding dimension of the action configurable
        self.fc_a = nn.Sequential(nn.Linear(action_dim, 16), nn.ReLU())
        self.fc_cat = nn.Sequential(
                        nn.Linear(self.encoder.output_dim + 16, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1))

    def forward(self, x, a):
        h1 = self.encoder(x)
        h2 = self.fc_a(a)
        cat = torch.cat([h1,h2], dim=1)
        out = self.fc_cat(cat)
        return out
    

# container for 2 Q networks.
class Critic_TD3(nn.Module):
    def __init__(self, se1, se2, a_dim):
        super(Critic_TD3, self).__init__()

        # Q1 architecture
        self.q1 = QNet_TD3(se1, a_dim)
        
        # Q2 architecture
        self.q2 = QNet_TD3(se2, a_dim)


    def forward(self, state, action):
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2

    def Q1(self, state, action):
        return self.q1(state, action)
 
