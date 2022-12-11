python -m torch.distributed.launch --nproc_per_node=4 --master_port=12233 --use_env run_train.py \
--diff_steps 1000 \
--lr 1e-4 \
--learning_steps 30000 \
--save_interval 5000 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 1024 \
--dataset qqp \
--data_dir datasets/QQP \
--vocab bert \
--model_name lstm \
--seq_len 128 \
--schedule_sampler lossaware \
--notes lstm-qqp \


