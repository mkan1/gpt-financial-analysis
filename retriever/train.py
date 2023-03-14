from tqdm import tqdm
import json
import os
from datetime import datetime
import time
import logging
from utils import read_examples, read_txt
from config import parameters as conf
from torch import nn
import torch
import torch.optim as optim
from transformers import BertTokenizer
from transformers import BertConfig

tokenizer = BertTokenizer.from_pretrained(conf.model_size)
model_config = BertConfig.from_pretrained(conf.model_size)

model_dir_name = conf.model_save_name + "_" + \
        datetime.now().strftime("%Y%m%d%H%M%S")
model_dir = os.path.join(conf.output_path, model_dir_name)
results_path = os.path.join(model_dir, "results")
saved_model_path = os.path.join(model_dir, "saved_model")
os.makedirs(saved_model_path, exist_ok=False)
os.makedirs(results_path, exist_ok=False)
log_file = os.path.join(results_path, 'log.txt')

op_list = read_txt(conf.op_list_file, log_file)
op_list = [op + '(' for op in op_list]
op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
const_list = read_txt(conf.const_list_file, log_file)
const_list = [const.lower().replace('.', '_') for const in const_list]
train_data, train_examples, op_list, const_list = \
    read_examples(input_path=conf.train_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file)

valid_data, valid_examples, op_list, const_list = \
    read_examples(input_path=conf.valid_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file)

test_data, test_examples, op_list, const_list = \
    read_examples(input_path=conf.test_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file)

print(train_data[0])