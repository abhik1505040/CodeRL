##
## Copyright (c) 2022, salesforce.com, inc.
## All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
## For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
##
# model_path="/media/rifat/HDD/CodeRL/exps/codet5-large_rl_bs1x4_lr2e-05/checkpoint-250000"
model_path="/media/rifat/HDD/CodeRL_exps/codet5-large_none_bs1x4_lr2e-05/checkpoint-293000"
tokenizer_path=models/codet5_tokenizer/
test_path=data/APPS/test/ 

start=0
end=5000
num_seqs_per_iter=5 
num_seqs=5
temp=0.6

# output_path=outputs/codes/
output_path=outputs/codes_actor/

CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_path $model_path \
    --tokenizer_path $tokenizer_path \
    --test_path $test_path \
    --output_path $output_path \
    -s $start -e $end \
    --num_seqs $num_seqs --num_seqs_per_iter $num_seqs_per_iter \
    --temperature $temp \