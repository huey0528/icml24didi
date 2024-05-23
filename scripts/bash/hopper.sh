for seed in 0 1 2; do
    python scripts/train_re.py --dataset hopper-medium-v2 --seed $seed
    python scripts/train_re.py --dataset hopper-medium-replay-v2 --seed $seed
    python scripts/train_re.py --dataset hopper-medium-expert-v2  --seed $seed
done