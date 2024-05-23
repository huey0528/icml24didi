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



policy = Policy(didi, dataset.normalizer, horizon, observation_action_dim)
disc = Disc(didi_experiment.disc, dataset.normalizer, horizon, observation_action_dim)

# action_experiment = utils.load_action_fn(args.logbase, args.dataset, args.action_loadpath, epoch=args.reward_epoch)# if not args.knowaction else None
action_model = None# action_experiment.ema
#---------------------------------- main loop ----------------------------------#



from diffuser.utils.logx import EpochLogger
# logger = EpochLogger(args.savepath)

vvv = True
v_output_dir = args.savepath + "/v"
if os.path.exists(v_output_dir):
    pass
else:
    os.makedirs(v_output_dir)

ROLLOUTS = []
# for i in [0,1,2,3,4,5]:
for i in range(args.skill_dim):

    observation = env.reset()
    ## observations for rendering
    rollout = [observation.copy()]

    skill_id = i
    skill = np.random.rand(skill_dim) * 0.
    skill[skill_id] = 1.
    skill = np.array(skill, dtype=np.float32)
    
    print(skill)

    cond = {}
    total_reward = 0
    T = int(env.max_episode_steps/2)
    for t in range(T):
        action, samples = policy(observation, skill, action_model)
        next_observation, reward, terminal, _ = env.step(action)
        div = 0#disc(next_observation, samples.observations[0][0], skill)
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        Escore = env.get_normalized_score(total_reward * env.max_episode_steps / (t+1))
        if t % 100 == 0:
            print(f'{skill_id}: t: {t}/{T} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | Escore: {Escore:.4f} | ')
            print('  ', div)
        if t%10==0: rollout.append(next_observation.copy())
        # if terminal: print(terminal)
        observation = next_observation

        if vvv and (t % 10000 == 0 or t == T-1 or terminal):
            fullpath = join(args.savepath, f'v/skillnum{args.skill_dim}_seed{args.seed}_{skill_id}_{t}.png')
            # if t == 0: renderer.composite(fullpath, samples.observations, )#ncol=1)
            # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)
            ## save rollout thus far
            # if t == T-1: renderer.composite(join(args.savepath, f'v/skillnum{args.skill_dim}_seed{args.seed}_{skill_id}_rollout.png'), np.array(rollout)[None], )#ncol=1)
            # renderer.render_rollout(join(args.savepath, f'v/{skill_id}_rollout.mp4'), rollout, fps=80)
            # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)
    ROLLOUTS.append(rollout)
    # logger.log_tabular('T', t)
    # logger.log_tabular('MaxT', env.max_episode_steps)
    # logger.log_tabular('r', np.round(reward, 2))
    # logger.log_tabular('R', np.round(total_reward, 2))
    # logger.log_tabular('score', np.round(score, 4))
    # logger.log_tabular('Escore', np.round(Escore, 4))
    # logger.dump_tabular()

renderer.composite(join(args.savepath, f'v/skillnum{args.skill_dim}_seed{args.seed}_rollout.png'), np.array(ROLLOUTS), )#ncol=1)
np.save(join(args.savepath, f'v/skillnum{args.skill_dim}_seed{args.seed}_rollout.npy'), np.array(ROLLOUTS))

# ## save result as a json file
# json_path = join(args.savepath, 'rollout.json')
# json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
#     'epoch_diffusion': diffusion_experiment.epoch}
# json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)


