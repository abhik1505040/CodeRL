import torch
import glob
import logging
import random
import fnmatch
import numpy as np
import gc
import os
from tqdm import tqdm 
from collections import Counter
import pickle as pkl 
import json, pdb 

from multiprocessing import Manager
from transformers import AutoTokenizer

import dataset_impl.utils as dsutils

# Reuse the CodeRL base implementation with minor changes
# until we decide to introduce major changes

class APPSBaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, problem_dirs, max_tokens, max_src_tokens):
        
        self.dataroot = dataroot
        self.problem_dirs = problem_dirs 

        self.max_tokens = max_tokens
        self.max_src_tokens = max_src_tokens

        self.samples = []           
        self.initialize()

        self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
        
    
    def initialize(self):
        all_samples = []
        skipped_problems = []
        
        print(f"Loading {len(self.problem_dirs)} problems from {self.dataroot}.")
        for problem_name in tqdm(self.problem_dirs):           
                
            question_fname = os.path.join(self.dataroot, problem_name, "question.txt")
            sols_fname = os.path.join(self.dataroot, problem_name, "solutions.json")
            io_path = os.path.join(self.dataroot, problem_name, "input_output.json")
 
            if (
                    not os.path.isfile(question_fname) or 
                    not os.path.isfile(sols_fname) or
                    not os.path.isfile(io_path) or 
                    len(json.load(open(io_path))["inputs"]) == 0):

                skipped_problems.append(problem_name)
                continue
                
            # Read the question description
            with open(question_fname, 'r') as f:
                question_str = f.read()
            
            starter_code = os.path.join(self.dataroot, problem_name, "starter_code.py")    
            if (os.path.isfile(starter_code)):
                answer_type = "\nUse Call-Based format\n"
                with open(starter_code, 'r') as f:
                    starter_code = f.read()
            else:
                answer_type = "\nUse Standard Input format\n"
                starter_code = ""

            sols_str_list = json.load(open(sols_fname, 'r'))
            gt_samples = self.load_gt_samples(
                sols_str_list, 
                answer_type, 
                starter_code, 
                question_str, 
                os.path.dirname(question_fname)
            )
            all_samples += gt_samples 
                    
        print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")
        print(f"Skipped {len(skipped_problems)} problems from {self.dataroot}.")
            
        self.samples = all_samples

    def load_gt_samples(self, sols, answer_type, starter_code, question_str, problem_path):
        samples = []
        
        for sol_str in sols:
            sol_str = dsutils.reindent_code(sol_str)
            sample = (question_str, starter_code, sol_str, answer_type, problem_path)
            samples.append(sample)
        
        return samples 
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        curr_q, curr_s, curr_a, curr_q_prefix, problem_path = self.samples[idx]
        sample = (curr_q[:150000], curr_s[:150000], curr_a[:150000], curr_q_prefix, problem_path)
        return self.tokenize(sample)
    
    def tokenize(self, sample):
        input_ids = []                        
        q_str, s_str, a_str, answer_type, problem_path = sample
        q_str =  "\nQUESTION:\n" + q_str + "\n" + s_str + "\n" + answer_type + "\nANSWER:\n"

        question_token_ids = self.tokenizer.encode(q_str, verbose=False)
        input_ids.extend(question_token_ids)
            
        # Sanity checks and padding 
        input_ids_max_len = self.max_src_tokens

        if len(input_ids) < input_ids_max_len: 
            new_input_ids = [self.tokenizer.eos_token_id] * input_ids_max_len
            new_input_ids[:len(input_ids)] = input_ids
            input_ids = new_input_ids 
                    
        # Cut off the excess
        input_ids = input_ids[:input_ids_max_len]
        out_sample = {
            "input_ids" : torch.LongTensor(input_ids),
            "problem_path": problem_path
        }
            
        return out_sample 


def ppo_data_collator(features):
    return {k: [f[k] for f in features] for k in features[0]}
    