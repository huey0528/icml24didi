import sys, os
sys.path.append(os.getcwd())

import json
import numpy as np
import torch
from os.path import join
import pdb
import einops

from diffuser.guides.skill_policy import Policy, Disc
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'hopper-medium-v2'
    config: str = 'config.locomotion'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('didi')
env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

didi_experiment = utils.load_didi(args.logbase, args.dataset, args.didi_loadpath, epoch=args.didi_epoch)
didi = didi_experiment.ema
dataset = didi_experiment.dataset
renderer = didi_experiment.renderer

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim
observation_action_dim = observation_dim + action_dim
horizon = dataset.horizon
batch_size = 1 #args.batch_size

skill_dim = dataset.skill_dim
skill_dt = dataset.skill_dt


ROLLOUTS = np.load(join(args.savepath, f'v/skillnum{args.skill_dim}_seed{args.seed}_rollout.npy'))

rollouts = ROLLOUTS#[:, ::5, :]

mine_render_kwargs = {
            'trackbodyid': 2,
            'distance': 4,
            'lookat': [2, 2, 0.5],
            'elevation': 0
        }

renderer.composite(join(args.savepath, f'v/skillnum{args.skill_dim}_seed{args.seed}_rollout_test.png'), rollouts, mine_render_kwargs=mine_render_kwargs)#ncol=1)







