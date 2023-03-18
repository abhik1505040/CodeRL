
# To disable the annoying warnings in terminal
export TOKENIZERS_PARALLELISM=true

python \
    train_ppo.py \
    --model_name "/media/rifat/HDD/CodeRL_exps/codet5-large_none_bs1x4_lr2e-05/checkpoint-293000" \
    --save_dir "exps/PPO_model" \
    --num_epochs 2 \
    --max_src_tokens 600 \
    --max_tgt_tokens 512 \
    --batch_size 4 \
    --optimize_cuda_cache \
    --log_with wandb