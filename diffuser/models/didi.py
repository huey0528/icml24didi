import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import einops

import diffuser.utils as utils

from .helpers import (
    Losses,
    apply_conditioning,
)

        
class DIDI(nn.Module):
    def __init__(self, policy, disc, normalizer, n_timesteps=100, horizon=128,
        reward_guidance_scale=1., loss_type='l2', action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.policy = policy
        self.disc = disc 
        
        self.normalizer = normalizer
        self.reward_guidance_scale = reward_guidance_scale

        self.action_dim = normalizer.action_dim
        self.state_dim = normalizer.observation_dim
        self.state_action_dim = self.action_dim + self.state_dim

        self.n_timesteps = n_timesteps
        self.horizon = horizon

        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)
    
    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.state_action_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        loss_weights[0, :] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def conditional_sample(self, cond, *args, **kwargs):
        device = cond[0].device
        batch_size = len(cond[0])
        traj = torch.zeros(batch_size, self.horizon, self.state_action_dim, device=device, dtype=torch.float32)
        # traj[:, :, self.action_dim: ] += einops.rearrange(cond[0], 'b s -> b 1 s')
        # traj[:, 1:, :self.action_dim] += einops.rearrange(torch.arange(1, self.horizon, device=device, dtype=torch.float32)/self.horizon, 'h -> 1 h 1')
        traj[:, :, self.action_dim: ] += einops.rearrange(cond[0], 'b s -> b 1 s')
        traj[:, 1:, :self.action_dim] += einops.rearrange(torch.arange(1, self.horizon, device=device, dtype=torch.float32)/self.horizon, 'h -> 1 h 1')

        skill = np.random.rand(self.disc.skill_dim)
        skills = []
        for i in range(batch_size):
            skill = np.zeros_like(skill)
            skill[i] = 1.
            skills.append(skill.copy())
        skills = np.array(skills, dtype=np.float32)
        skills = utils.to_torch(skills, device=device, dtype=torch.float32)

        x = self.policy(traj, skills)

        x = apply_conditioning(x, cond, self.action_dim)
        return x

    #------------------------------------------ training ------------------------------------------#

    def _skill_guide(self, skills, reward_model, rgs):
        def guide(traj):
            with torch.enable_grad():
                if len(traj.shape) == 2: traj = einops.rearrange(traj, 'h t -> 1 h t')
                b = traj.shape[0]
                ## shape: (batch_size*horizon)  * (action_dim+state_dim)
                traj = traj.detach().requires_grad_(True)
                re = reward_model.get_return(traj) * rgs - F.mse_loss(self.disc(traj), skills)
                return torch.autograd.grad(re, traj)[0].reshape(b, -1, self.state_action_dim)
        return guide
        

    def loss(self, diffusion_model, reward_model, conditions, skills):

        batch_size = skills.shape[0]
        noise = torch.rand(batch_size, self.horizon, self.state_action_dim, device=skills.device, dtype=torch.float32)
        
        traj = torch.zeros(batch_size, self.horizon, self.state_action_dim, device=skills.device, dtype=torch.float32)
        traj[:, :, self.action_dim: ] += einops.rearrange(conditions[0], 'b s -> b 1 s')
        traj[:, 1:, :self.action_dim] += einops.rearrange(torch.arange(1, self.horizon, device=skills.device, dtype=torch.float32)/self.horizon, 'h -> 1 h 1')

        x_start = self.policy(traj, skills)

        x_start_c = apply_conditioning(x_start, conditions, self.action_dim).detach()
        t = torch.randint(0, self.n_timesteps-0, (batch_size,), device=skills.device).long()
        if np.random.rand() > 0.50:
            pass
        else:
            x_start_c = torch.rand(batch_size, self.horizon, self.state_action_dim, device=skills.device, dtype=torch.float32)
            t = torch.randint(self.n_timesteps-5, self.n_timesteps, (batch_size,), device=skills.device).long()
        
        x_noisy = diffusion_model.q_sample(x_start=x_start_c, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, conditions, self.action_dim)
        x_recon = diffusion_model.model(x_noisy, conditions, t)
        x_recon = apply_conditioning(x_recon, conditions, self.action_dim)

        # guide_fun = self._skill_guide(skills, reward_model, self.reward_guidance_scale)
        # x_recon = diffusion_model(conditions, reward_guidance=guide_fun, verbose=False)

        assert noise.shape == x_recon.shape

        # TODO: depends on the diffusion training setup
        assert diffusion_model.predict_epsilon == False
        loss_diffusion, _ = self.loss_fn(x_start, x_recon.detach())
        # loss_diffusion, _ = F.mse_loss(x_start, x_recon.detach())

        loss_reward = -1. * reward_model.get_return(x_start) * self.reward_guidance_scale
        # loss_diversity = F.mse_loss(self.disc(x_start[:, 1, self.action_dim:]), skills)
        # loss_diversity = F.mse_loss(self.disc(x_start[:, :, self.action_dim+5:self.action_dim+6].mean(1)), skills)
        # loss_diversity = F.mse_loss(self.disc(x_start[:, :, self.action_dim+5:self.action_dim+6].mean(1)), skills)
        loss_diversity = F.mse_loss(self.disc(x_start), skills)

        loss = loss_diffusion * 10 + loss_reward  + loss_diversity *50

        info = {'diff': loss_diffusion.detach().cpu().numpy(), 
                'rew': loss_reward.detach().cpu().numpy(), 
                'div': loss_diversity.detach().cpu().numpy()}

        return loss, info
    
    def loss_disc(self, diffusion_model, reward_model, conditions, skills):

        batch_size = skills.shape[0]
        
        traj = torch.zeros(batch_size, self.horizon, self.state_action_dim, device=skills.device, dtype=torch.float32)
        traj[:, :, self.action_dim: ] += einops.rearrange(conditions[0], 'b s -> b 1 s')
        traj[:, 1:, :self.action_dim] += einops.rearrange(torch.arange(1, self.horizon, device=skills.device, dtype=torch.float32)/self.horizon, 'h -> 1 h 1')

        x_start = self.policy(traj, skills).detach()
        x_start = x_start + (torch.rand_like(x_start, device=skills.device)-1.0) * 2 * 0.2
        x_start = x_start.detach()
        
        # loss_diversity = F.mse_loss(self.disc(x_start[:, 1, self.action_dim:]), skills)
        # loss_diversity = F.mse_loss(self.disc(x_start[:, :, self.action_dim+5:self.action_dim+6].mean(1)), skills)
        # loss_diversity = F.mse_loss(self.disc(x_start[:, :, self.action_dim+5:self.action_dim+6].mean(1)), skills)
        loss_diversity = F.mse_loss(self.disc(x_start), skills) *5
        

        return loss_diversity, {}

    def forward(self, cond, state, skill):
        return 1/0
    
