#
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

# Run code in debugging mode (without deepspeed) 
python \
    train.py \
    --batch-size-per-replica 1 --grad-acc-steps 4 \
    --epochs 10 --lr 2e-5 \
    --save-freq 1000 --log-freq 10 --save_total_limit 20 \
    --fp16 \
    --tuning_mode rl --model codet5-large \
    --model_path /media/rifat/HDD/CodeRL_exps/codet5-large_none_bs1x4_lr2e-05/checkpoint-293000 \
    --relative_returns
