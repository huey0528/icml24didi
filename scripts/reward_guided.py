import sys, os
sys.path.append(os.getcwd())

import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.plan_policy import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'hopper-medium-v2'
    config: str = 'config.locomotion'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')
env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)
diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

reward_experiment = utils.load_reward_fn(args.logbase, args.dataset, args.reward_loadpath, epoch=args.reward_epoch)  if not args.knowr else None

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

policy = Policy(diffusion_experiment.ema, 
                reward_experiment.ema  if not args.knowr else utils.KReward(env, action_dim), 
                dataset.normalizer, 
                reward_guidance_scale=args.rgs)

#---------------------------------- main loop ----------------------------------#

observation = env.reset()
## observations for rendering
rollout = [observation.copy()]

from diffuser.utils.logx import EpochLogger
logger = EpochLogger(args.savepath)

vvv = False

cond = {}
cond[0] = observation
# cond[1] = observation
# cond[2] = observation
total_reward = 0
for t in range(env.max_episode_steps):
    cond[0] = observation#cond[1].copy()
    # cond[1] = cond[2].copy()
    # cond[2] = observation
    action, samples = policy(cond, batch_size=args.batch_size)
    next_observation, reward, terminal, _ = env.step(action)

    total_reward += reward
    score = env.get_normalized_score(total_reward)
    Escore = env.get_normalized_score(total_reward * env.max_episode_steps / (t+1))
    print(
        f't: {t}/{env.max_episode_steps} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | Escore: {Escore:.4f} | '
    )
    rollout.append(next_observation.copy())
    if vvv and (t % 10 == 0 or t == env.max_episode_steps-1):
        fullpath = join(args.savepath, f'{t}.png')
        if t == 0: renderer.composite(fullpath, samples.observations, )#ncol=1)
        # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)
        ## save rollout thus far
        renderer.composite(join(args.savepath, f'{t}_rollout.png'), np.array(rollout)[None], )#ncol=1)
    if terminal:
        # break
        print(terminal)
    observation = next_observation

    logger.log_tabular('T', t)
    logger.log_tabular('MaxT', env.max_episode_steps)
    logger.log_tabular('r', np.round(reward, 2))
    logger.log_tabular('R', np.round(total_reward, 2))
    logger.log_tabular('score', np.round(score, 4))
    logger.log_tabular('Escore', np.round(Escore, 4))
    logger.dump_tabular()

# ## save result as a json file
# json_path = join(args.savepath, 'rollout.json')
# json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
#     'epoch_diffusion': diffusion_experiment.epoch}
# json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)


