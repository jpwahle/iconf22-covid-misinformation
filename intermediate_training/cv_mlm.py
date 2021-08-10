# All imports
import os
import re
import json
import torch
import tokenizers
import transformers

import pandas as pd

from tqdm import tqdm
from pathlib import Path
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

################ Setup ##############
#####################################
#If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.   
    # gpu_number = 3 
    # gpu_name = "cuda:" + str(gpu_number)
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    
    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def get_model_tokenizer(model_name, max_seq_len):
    # BART
    # Using the default config
    model = transformers.AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    # Tokenizer
    ## Found that new tokenizer doesn't work well
    # print('Using new tokenizer')
    # tokenizer = transformers.AutoTokenizer.from_pretrained("models/COVID", max_len=max_seq_len)
    ## Using the default tokenizer
    print("Using default tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_len=max_seq_len)
    return model, tokenizer

# Dataset reading
lm_data_dir = "lm_data"
train_path = os.path.join(lm_data_dir,"train.txt")
eval_path = os.path.join(lm_data_dir,"eval.txt")

# Getting model and tokenizer
#model_name = 'microsoft/deberta-base'
model_name = 'allenai/longformer-base-4096'
max_seq_len = 1024
model, tokenizer = get_model_tokenizer(model_name, max_seq_len)

output_dir_name = "models/COVID_longformer"

# Initially block size was 32
print('Begining Data Collator')
dataset = transformers.LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = "lm_data/train.txt", #should be train
    block_size = 128,
)

data_collator = transformers.DataCollatorForLanguageModeling(
    tokenizer = tokenizer, mlm = True, mlm_probability = 0.15
)
print('Finished data collator')
# Initially batch size was 32
training_args = transformers.TrainingArguments(
    output_dir = output_dir_name,
    overwrite_output_dir = True,
    num_train_epochs = 5,
    per_gpu_train_batch_size = 4,
    save_steps = 10000,
    save_total_limit = 2,
)

trainer = transformers.Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = dataset,
)
print('In training stage')
# Training
trainer.train()
# Sacing model
trainer.save_model(output_dir_name)
