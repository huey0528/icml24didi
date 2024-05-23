import sys, os
sys.path.append(os.getcwd())

import diffuser.utils as utils


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('reward')


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
)

# render_config = utils.Config(
#     args.renderer,
#     savepath=(args.savepath, 'render_config.pkl'),
#     env=args.dataset,
# )

dataset = dataset_config()
# renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    input_dim=observation_dim + action_dim,
    hidden_dim=args.hidden_dim,
    device=args.device,
)

reward_config = utils.Config(
    args.reward,
    savepath=(args.savepath, 'reward_config.pkl'),
    # horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    # n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    # clip_denoised=args.clip_denoised,
    # predict_epsilon=args.predict_epsilon,
    ## loss weighting
    # action_weight=args.action_weight,
    # loss_weights=args.loss_weights,
    # loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size // args.horizon,
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

model = model_config()

reward = reward_config(model)

trainer = trainer_config(reward, dataset, None)

if args.load:
    load_epoch = 0
    load_epoch = utils.serialization.get_latest_epoch(trainer.logdir.split('/'))
    print(f'\n[ utils/serialization ] (Reward) Loading model epoch: {load_epoch}\n')
    trainer.load(load_epoch)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model, topk=5)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = reward.loss(*batch)
loss.backward()
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

from diffuser.utils.logx import EpochLogger
logger = EpochLogger(args.savepath)

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    loss, infos = trainer.train(n_train_steps=args.n_steps_per_epoch)
    logger.log_tabular('Epoch', i)
    logger.log_tabular('Loss', loss.cpu().detach().numpy())
    logger.dump_tabular()

