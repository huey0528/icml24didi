import os
import pickle
import glob
import torch
import pdb

from collections import namedtuple

DiffusionExperiment = namedtuple('Diffusion', 'dataset renderer model diffusion ema trainer epoch')
RewardExperiment = namedtuple('Reward', 'dataset renderer model reward ema trainer epoch')
DidiExperiment = namedtuple('Didi', 'dataset renderer model disc didi ema trainer epoch')
ActionExperiment = namedtuple('Action', 'dataset renderer model action ema trainer epoch')

def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False

def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    print(loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    print(config)
    return config

def load_diffusion(*loadpath, epoch='latest', device='cuda:0'):
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    render_config = load_config(*loadpath, 'render_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    diffusion_config = load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    dataset = dataset_config()
    renderer = render_config()
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] (Diffusion) Loading model epoch: {epoch}\n')

    trainer.load(epoch)

    return DiffusionExperiment(dataset, renderer, model, diffusion, trainer.ema_model, trainer, epoch)

def load_reward_fn(*loadpath, epoch='latest', device='cuda:0'):
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    reward_config = load_config(*loadpath, 'reward_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    dataset = dataset_config()
    model = model_config()
    reward = reward_config(model)
    trainer = trainer_config(reward, dataset, None)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] (Reward) Loading model epoch: {epoch}\n')

    trainer.load(epoch)

    return RewardExperiment(dataset, None, model, reward, trainer.ema_model, trainer, epoch)

def load_didi(*loadpath, epoch='latest', device='cuda:0'):
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    render_config = load_config(*loadpath, 'render_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    disc_config = load_config(*loadpath, 'disc_config.pkl')
    didi_config = load_config(*loadpath, 'didi_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    dataset = dataset_config()
    renderer = render_config()
    model = model_config()
    disc = disc_config()
    didi = didi_config(model, disc, dataset.normalizer)
    trainer = trainer_config(didi, dataset, renderer)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] (Didi) Loading model epoch: {epoch}\n')

    trainer.load(epoch)

    return DidiExperiment(dataset, renderer, model, disc, didi, trainer.ema_model, trainer, epoch)


def load_action_fn(*loadpath, epoch='latest', device='cuda:0'):
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    action_config = load_config(*loadpath, 'action_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    dataset = dataset_config()
    model = model_config()
    action = action_config(model)
    trainer = trainer_config(action, dataset, None)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] (Action) Loading model epoch: {epoch}\n')

    trainer.load(epoch)

    return ActionExperiment(dataset, None, model, action, trainer.ema_model, trainer, epoch)