for seed in 0 1 2; do
    python scripts/train_re.py --dataset walker2d-medium-v2 --seed $seed
    python scripts/train_re.py --dataset walker2d-medium-replay-v2 --seed $seed
    python scripts/train_re.py --dataset walker2d-medium-expert-v2  --seed $seed
done