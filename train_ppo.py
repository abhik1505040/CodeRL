import logging
import os
import sys
import json
from typing import Optional
from transformers import HfArgumentParser, AutoTokenizer
from dataclasses import dataclass, make_dataclass, field, asdict
import inspect
 
from tqdm import tqdm

from generate import generate_compiler_result
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed

import transformers
import multiprocessing
import torch

from dataset_impl.ppo_dataset import APPSBaseDataset, ppo_data_collator
import dataset_impl.utils as dsutils

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)

@dataclass
class GenerationArguments:
    # Max generation length comes from DataTrainingArguments
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether to sample"}
    )
    temperature: float = field(
        default=0.6,
        metadata={"help": "Sampling temperature"}
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "nucleus sampling value"}
    )

@dataclass
class DataTrainingArguments:

    save_dir: str = field(
        metadata={"help": 'path to save trained model checkpoints'}
    )
    num_epochs: int = field(
        default=1,
        metadata={'help': "No of overall training epochs"}
    )
    train_path: str = field(
        default='data/APPS/train/',
        metadata={"help": 'path to training data'}
    )
    max_src_tokens: int = field(
        default=600,
        metadata={'help': "Max no of input tokens"}
    )
    max_tgt_tokens: int = field(
        default=512,
        metadata={'help': "Max no of output tokens"}
    )

def generate_rewards(program_iterator, timeout=10):
    # initialize with runtime errors
    compiler_results = multiprocessing.Manager().list([-1] * len(program_iterator))
    # no. of total concurrent processes.
    # adding some extra since a there's a little io dependency rn
    process_count = multiprocessing.cpu_count() + 3

    def _get_chunks(iterator, chunk_size):
        return [iterator[i: i + chunk_size] for i in range(0, len(iterator), chunk_size)]
    
    def _populate_result(idx, solution, problem_path):
        result = generate_compiler_result(solution, problem_path)
        compiler_results[idx] = result

    cur_idx = 0
    for chunk in _get_chunks(program_iterator, process_count):
        processes = []
        # start all processes
        for solution, problem_path in chunk:
            process = multiprocessing.Process(
                target=_populate_result, 
                args=(cur_idx, solution, problem_path)
            )
            process.start()
            processes.append(process)
            cur_idx += 1
        
        # cleanup the processes, the timeout here is not
        # exactly what it's supposed to be. but we only enforce
        # this for graceful termination anyway. So, who cares! 
        for process in processes:
            process.join(timeout=timeout)
            if process.is_alive(): process.kill()

    rewards = [dsutils.get_reward_from_error_type(dsutils.get_error_type(k))
                    for k in compiler_results]
    
    return rewards

def save_checkpoint(trainer, save_dir):
    # simplified checkpoint saving without creating model cards
    logger.info(f"Saving checkpoint at {save_dir}")
    trainer.accelerator.unwrap_model(trainer.model).save_pretrained(save_dir)
    trainer.tokenizer.save_pretrained(save_dir)
        

def run_training(data_args, ppo_args, generation_args, train_data):
    ppo_config = PPOConfig(**asdict(ppo_args))
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(ppo_config.model_name)
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=train_data,
        data_collator=ppo_data_collator
    )

    generation_kwargs = asdict(generation_args)
    generation_kwargs.update({"max_length": data_args.max_tgt_tokens})
    epoch_iterator = trainer.dataloader
    
    # write all args to disk
    all_args = asdict(data_args) | asdict(ppo_args) | asdict(generation_args)
    args_path = os.path.join(data_args.save_dir, "args.json")
    logger.info(f"Saving input args as {args_path}")

    with open(args_path, 'w') as f:
        json.dump(all_args, f, indent=4)

    # Minimal training loop
    for epoch in range(data_args.num_epochs):
        for batch in tqdm(epoch_iterator, desc=f"Epoch-{epoch}"):
            # Step 1: Generate response using the current policy
            query_tensors = batch["input_ids"]
            response_tensors = []
            
            for query in query_tensors:
                response = trainer.generate(query, **generation_kwargs)
                response_tensors.append(response.squeeze())

            batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) 
                                    for r in response_tensors]
            
            # Step 2: Generate Rewards
            rewards = generate_rewards(list(zip(batch["response"], batch["problem_path"])))
            rewards = [torch.tensor(k, dtype=torch.float).to(trainer.accelerator.device) for k in rewards]

            # Step 3: Run PPO Step
            stats = trainer.step(query_tensors, response_tensors, rewards)
            trainer.log_stats(stats, batch, rewards)
            
        if trainer.accelerator.is_main_process:
            checkpoint_directory = os.path.join(data_args.save_dir, f"epoch-{epoch}")
            save_checkpoint(trainer, checkpoint_directory)

    

def get_dataset(args): 
    fnames = os.listdir(args.train_path)
    
    train_data = APPSBaseDataset(
        dataroot=args.train_path, 
        problem_dirs=fnames,
        max_tokens=args.max_tgt_tokens,
        max_src_tokens=args.max_src_tokens       
    )

    return train_data


def main():
    # a simple and 'lazy' way to enable cmd line flags for
    # all non composite arguments of PPOConfig
    ppo_arguments = [(k, v.annotation, field(default=v.default)) 
                        for k, v in inspect.signature(PPOConfig).parameters.items()
                            if not isinstance (v.default, (list, dict))]

    PPOArguments = make_dataclass('PPOArguments', ppo_arguments)

    parser = HfArgumentParser((PPOArguments, DataTrainingArguments, GenerationArguments))
    ppo_args, data_args, generation_args = parser.parse_args_into_dataclasses()
    
    # Explicitly overwrite some options
    ppo_args.remove_unused_columns = False

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"PPO parameters: {ppo_args}")
    logger.info(f"Generation parameters: {generation_args}")

    os.makedirs(data_args.save_dir, exist_ok=True)

    # Load dataset 
    train_data = get_dataset(data_args)

    # Load and train model; save model checkpoints 
    run_training(data_args, ppo_args, generation_args, train_data)


if __name__ == "__main__":
    main()
