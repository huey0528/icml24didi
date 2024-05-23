import numpy as np
import torch
from torch import nn

from .helpers import (
    Losses,
)

class DReward(nn.Module):
    def __init__(self, model, observation_dim, action_dim, n_timesteps=1000,
        loss_type='value_l2',
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.model = model

        self.loss_fn = Losses[loss_type]()
        

    #------------------------------------------ sampling ------------------------------------------#

    def get_return(self, traj):
        # assert len(traj.shape) == 3
        # assert traj.shape[0] == 1
        sa_dim = traj.shape[-1]
        traj = traj.reshape(-1, sa_dim)
        return self.model(traj).mean()

    #------------------------------------------ training ------------------------------------------#

    def loss(self, sa, rew):
        sa_dim = sa.shape[-1]
        sa = sa.reshape(-1, sa_dim)
        rew = rew.reshape(-1, 1)

        rew_pre = self.model(sa)
        loss, info = self.loss_fn(rew_pre, rew)
        
        return loss, {}

    def forward(self, cond, *args, **kwargs):
        return 1/0

