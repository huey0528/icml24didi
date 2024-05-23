import numpy as np
import torch
from torch import nn

from .helpers import (
    Losses,
)

class DAction(nn.Module):
    def __init__(self, model, observation_dim, action_dim, n_timesteps=1000,
        loss_type='value_l2',
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.model = model

        self.loss_fn = Losses[loss_type]()
        

    #------------------------------------------ sampling ------------------------------------------#

    def get_action(self, state, next_state):
        ss = torch.cat((state, next_state), 1)
        return self.model(ss)

    #------------------------------------------ training ------------------------------------------#

    def loss(self, ss, action):
        ss = ss.reshape(-1, ss.shape[-1])
        action = action.reshape(-1, action.shape[-1])

        action_pre = self.model(ss)
        # loss, info = self.loss_fn(action_pre, action)
        loss = nn.functional.mse_loss(action_pre, action)
        
        return loss, {}

    def forward(self, cond, *args, **kwargs):
        return 1/0

