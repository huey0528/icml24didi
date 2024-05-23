import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('seed', 'S'),
]

reward_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    # ('n_diffusion_steps', 'T'),
    ('seed', 'S'),
]

plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('seeddi', 'Sdi'),
    ('seedre', 'Sre'),
    ('seed', 'S'),
    ##
    ('conditional', 'cond'),
]

didi_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('skill_dim', 'sdm'),
    ('seed', 'S'),
    ##
    ('conditional', 'cond'),
]

action_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('seed', 'S'),
]


base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 128,
        'n_diffusion_steps': 100,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.MuJoCoRenderer',

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'SafeLimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 1000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 128,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,#1000
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': 314,

        'load': False,
    },

    'reward': {
        ## model
        'model': 'models.RewardNet',
        'reward': 'models.DReward',
        'horizon': 128,
        'hidden_dim': 256,
        # 'n_diffusion_steps': 100,
        # 'action_weight': 10,
        # 'loss_weights': None,
        # 'loss_discount': 1,
        # 'predict_epsilon': False,
        # 'dim_mults': (1, 4, 8),
        # 'renderer': 'utils.MuJoCoRenderer',

        ## dataset
        'loader': 'datasets.StateActionDataset',
        'normalizer': 'SafeLimitsNormalizer',
        'preprocess_fns': [],
        # 'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'reward/',
        'exp_name': watch(reward_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': int(5e5),
        'batch_size': 1024,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'log_freq': 1000,
        'save_freq': 10000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': 314,

        'load': False,
    },

    'plan': {
        'batch_size': 1,
        'device': 'cuda',

        ## diffusion model
        'horizon': 128,
        'n_diffusion_steps': 100,
        'normalizer': 'SafeLimitsNormalizer',
        'seed': 314,

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        ##
        'knowr': False,
        'rgs': 0.1,

        ## loading
        'seeddi': 314,
        'horizondi': 128,
        'diffusion_loadpath': 'f:diffusion/H{horizondi}_T{n_diffusion_steps}_S{seeddi}',
        'diffusion_epoch': 'latest',

        'seedre': 314,
        'horizonre':128,
        'reward_loadpath': 'f:reward/H{horizonre}_S{seedre}',
        'reward_epoch': 'latest',
    },

    'didi': {
        ## model
        'model': 'models.SkillNet',
        'didi': 'models.DIDI',
        'disc': 'models.DiscNet',
        'horizon': 128,
        'n_diffusion_steps': 100,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.MuJoCoRenderer',
        'hidden_dim': 256,
        'skill_dt': True,
        'skill_dim': 6,
        'disc_dims': None, #[5]
        'disc_mode': 0,
        'knowr': False,
        'rgs': 1.0,

        ## dataset
        'loader': 'datasets.SkillDataset',
        'normalizer': 'SafeLimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'didi/',
        'exp_name': watch(didi_args_to_watch),

        ## training
        'n_steps_per_epoch': 1000,
        'loss_type': 'l2',
        'n_train_steps': 1e5,
        'batch_size': 256,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'log_freq': 100,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': 314,

        ## loading
        'seeddi': 314,
        'horizondi': 128,
        'diffusion_loadpath': 'f:diffusion/H{horizondi}_T{n_diffusion_steps}_S{seeddi}',
        'diffusion_epoch': 'latest',

        'seedre': 314,
        'horizonre':128,
        'reward_loadpath': 'f:reward/H{horizonre}_S{seedre}',
        'reward_epoch': 'latest',

        'seedact': 314,
        'horizonact':128,
        'action_loadpath': 'f:action/H{horizonact}_S{seedact}',
        'action_epoch': 'latest',
        
        'didi_loadpath': 'f:didi/H{horizon}_T{n_diffusion_steps}_sdm{skill_dim}_S{seed}',
        'didi_epoch': 'latest',

        'load': False,
    },

    'action': {
        ## model
        'model': 'models.ActionNet',
        'action': 'models.DAction',
        'horizon': 128,
        'hidden_dim': 256,

        ## dataset
        'loader': 'datasets.StateStateActionDataset',
        'normalizer': 'SafeLimitsNormalizer',
        'preprocess_fns': [],
        # 'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'action/',
        'exp_name': watch(action_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': int(5e5),
        'batch_size': 1024,
        'learning_rate': 1e-3,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'log_freq': 1000,
        'save_freq': 10000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': 314,

        'load': False,
    },

}

#------------------------ overrides ------------------------#

## put environment-specific overrides here

test_v2 = {
    'diffusion': {
        'normalizer': 'SafeLimitsNormalizer',
    },
    'reward': {
        'normalizer': 'SafeLimitsNormalizer',
    },
    'didi': {
        'normalizer': 'SafeLimitsNormalizer',
    },
}



