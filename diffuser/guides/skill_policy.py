from collections import namedtuple
# import numpy as np
import torch
import torch.nn.functional as F
import einops
import pdb

import diffuser.utils as utils
# from diffusion.datasets.preprocessing import get_policy_preprocess_fn

Trajectories = namedtuple('Trajectories', 'actions observations')
# GuidedTrajectories = namedtuple('GuidedTrajectories', 'actions observations value')

class Policy:

    def __init__(self, didi, normalizer, horizon, observation_action_dim):
        self.didi_model = didi
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim
        self.horizon =horizon
        self.observation_action_dim = observation_action_dim


    @property
    def device(self):
        parameters = list(self.didi_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
        

    def __call__(self, observation, skill, action_model=None, batch_size=1):
        skill = utils.to_torch(skill, device=self.device, dtype=torch.float32)
        skill = einops.rearrange(skill, 'd -> 1 d')

        conditions = {0: observation}
        conditions = self._format_conditions(conditions, batch_size)

        # traj = torch.zeros(batch_size, self.horizon, self.observation_action_dim, device=self.device, dtype=torch.float32)
        # traj[:, :1, self.action_dim: ] = einops.rearrange(conditions[0], 'b s -> b 1 s')
        traj = torch.zeros(batch_size, self.horizon, self.observation_action_dim, device=self.device, dtype=torch.float32)
        traj[:, :, self.action_dim: ] += einops.rearrange(conditions[0], 'b s -> b 1 s')
        traj[:, 1:, :self.action_dim] += einops.rearrange(torch.arange(1, self.horizon, device=self.device, dtype=torch.float32)/self.horizon, 'h -> 1 h 1')

        sample = self.didi_model.policy(traj, skill)
        actions_from_action_model = action_model.get_action(sample[0, :-1, self.action_dim:], sample[0, 1:, self.action_dim:]) if action_model is not None else None

        sample = utils.to_np(sample)
        actions_from_action_model = utils.to_np(actions_from_action_model)  if action_model is not None else None

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        actions_from_action_model = self.normalizer.unnormalize(actions_from_action_model, 'actions')  if action_model is not None else None
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]
        # action = actions_from_action_model[0]

        # if debug:
        normed_observations = sample[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)

        trajectories = Trajectories(actions, observations)
        return action, trajectories
        # else:
        #     return action


class Disc:

    def __init__(self, disc, normalizer, horizon, observation_action_dim):
        self.disc = disc
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim
        self.horizon =horizon
        self.observation_action_dim = observation_action_dim


    @property
    def device(self):
        parameters = list(self.disc.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
        

    def __call__(self, observation, sample, skill, next_observation=None, next_sample=None, batch_size=1):
        observation = self.normalizer.normalize(observation, 'observations')
        sample = self.normalizer.normalize(sample, 'observations')

        observation = utils.to_torch(observation, device=self.device, dtype=torch.float32)
        observation = einops.rearrange(observation, 'd -> 1 d')
        sample = utils.to_torch(sample, device=self.device, dtype=torch.float32)
        sample = einops.rearrange(sample, 'd -> 1 d')

        skill = utils.to_torch(skill, device=self.device, dtype=torch.float32)
        skill = einops.rearrange(skill, 'd -> 1 d')
        

        # div0 = F.mse_loss(self.disc(observation), skill).detach().cpu().numpy().item()
        # div1 = F.mse_loss(self.disc(sample), skill).detach().cpu().numpy().item()
        div0 = F.mse_loss(self.disc(observation), skill).detach().cpu().numpy().item()
        div1 = F.mse_loss(self.disc(sample), skill).detach().cpu().numpy().item()
        

        div0 =round(div0, 3)
        div1 =round(div1, 3)
        
        return [div0, div1]