import sys, os
sys.path.append(os.getcwd())

import diffuser.utils as utils

from diffuser.models import DIDI
import diffuser.datasets as datasets


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('didi')

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)
diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

reward_experiment = utils.load_reward_fn(args.logbase, args.dataset, args.reward_loadpath, epoch=args.reward_epoch) if not args.knowr else None

assert args.horizon == diffusion_experiment.ema.horizon
assert args.n_diffusion_steps == diffusion_experiment.ema.n_timesteps

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    skill_dt=args.skill_dt,
    skill_dim=args.skill_dim,
)


render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()


observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    skill_dim=args.skill_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

disc_config = utils.Config(
    args.disc,
    savepath=(args.savepath, 'disc_config.pkl'),
    state_dim=observation_dim,
    action_dim=action_dim,
    skill_dim=args.skill_dim,
    hidden_dim=args.hidden_dim,
    disc_dims=args.disc_dims,
    disc_mode=args.disc_mode,
    device=args.device,
)

didi_config = utils.Config(
    args.didi,
    savepath=(args.savepath, 'didi_config.pkl'),
    loss_type=args.loss_type,
    ## loss weighting
    n_timesteps=args.n_diffusion_steps, 
    horizon=args.horizon,
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    reward_guidance_scale=args.rgs,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    log_freq=args.log_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

policy = model_config()
disc = disc_config()

didi = didi_config(policy, disc, dataset.normalizer)
                

trainer = trainer_config(didi, dataset, renderer,
                        didi_enable=True, 
                        didi_diffusion=diffusion_experiment.ema, 
                        didi_reward= reward_experiment.ema  if not args.knowr else utils.KReward(env, action_dim))

if args.load:
    load_epoch = 0
    load_epoch = utils.serialization.get_latest_epoch(trainer.logdir.split('/'))
    print(f'\n[ utils/serialization ] (DIDI) Loading model epoch: {load_epoch}\n')
    trainer.load(load_epoch)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(policy, topk=5)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = didi.loss(diffusion_experiment.ema, reward_experiment.ema  if not args.knowr else utils.KReward(env, action_dim), *batch)
loss.backward()
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

from diffuser.utils.logx import EpochLogger
logger = EpochLogger(args.savepath)

for epoch in range(n_epochs):
    print(f'Epoch {epoch} / {n_epochs} | {args.savepath}')
    loss, infos = trainer.train(n_train_steps=args.n_steps_per_epoch)
    logger.log_tabular('Epoch', epoch)
    logger.log_tabular('Loss', loss.cpu().detach().numpy())
    logger.log_tabular('Diff', infos['diff'])
    logger.log_tabular('Rew', infos['rew'])
    logger.log_tabular('Div', infos['div'])
    logger.dump_tabular()



    import numpy as np
    from os.path import join
    from diffuser.guides.skill_policy import Policy, Disc

    skill_dim = args.skill_dim
    policy_temp = Policy(trainer.model, dataset.normalizer, args.horizon, observation_dim+action_dim)

    vvv = True
    v_output_dir = args.savepath + "/v"
    if not os.path.exists(v_output_dir):
        os.makedirs(v_output_dir)

    ROLLOUTS = []
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
        T = int(env.max_episode_steps/2.0)
        for t in range(T):
            action, samples = policy_temp(observation, skill)
            next_observation, reward, terminal, _ = env.step(action)
            div = 0#disc(next_observation, samples.observations[0][0], skill)
            total_reward += reward
            score = env.get_normalized_score(total_reward)
            Escore = env.get_normalized_score(total_reward * env.max_episode_steps / (t+1))
            # if t%10==0: 
            rollout.append(next_observation.copy())
            # if terminal: print(terminal)
            observation = next_observation
            
        ROLLOUTS.append(rollout)
        
    renderer.composite(join(args.savepath, f'v/skillnum{args.skill_dim}_seed{args.seed}_rollout_{int(epoch*args.n_steps_per_epoch)}.png'), np.array(ROLLOUTS)[:,::10,:], )#ncol=1)
    np.save(join(args.savepath, f'v/skillnum{args.skill_dim}_seed{args.seed}_rollout_{int(epoch*args.n_steps_per_epoch)}.npy'), np.array(ROLLOUTS))

