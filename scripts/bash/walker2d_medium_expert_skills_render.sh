for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    for j in 3 5; do
        python scripts/skill_guided.py --dataset walker2d-medium-expert-v2 --horizondi 64 --seeddi 6 --horizonre 64 --seedre 6 --horizon 64 --seed $i --use_padding False --rgs 0  --sample_freq 1000 --skill_dim $j --disc_dims [$i] --knowr True --action_weight 5 --n_train_steps 10000
    done
done