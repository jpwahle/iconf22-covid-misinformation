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
#Force set cpu
#device = torch.device("cpu")
# Dataset reading
lm_data_dir = "lm_data"
train_path = os.path.join(lm_data_dir,"train.txt")
eval_path = os.path.join(lm_data_dir,"eval.txt")

# Tokenizer
# For now - Not using this
tokenizer = tokenizers.implementations.ByteLevelBPETokenizer(
    'models/COVID/vocab.json', 'models/COVID/merges.txt'
)
tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
# RobertA
config = transformers.RobertaConfig(
#    vocab_size = 5000,
    vocab_size = 50265,
    max_position_embeddings = 514,
    num_attention_heads = 12,
    num_hidden_layers = 6,
    type_vocab_size = 1,
)
#model = transformers.RobertaForMaskedLM(config).to(device)
#model = transformers.RobertaForMaskedLM.from_pretrained('roberta-base', config = config).to(device)
# Using the default config
model = transformers.RobertaForMaskedLM.from_pretrained('roberta-base').to(device)
#model.cuda()
#tokenizer = transformers.RobertaTokenizerFast.from_pretrained("models/COVID")
print('Using new tokenizer')
tokenizer = transformers.RobertaTokenizer.from_pretrained("models/COVID", max_len=512)
## Using the default tokenizer
# print("Using default tokenizer")
# tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base', max_len=512)
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
    output_dir = "models/COVID_RoBERTa",
    overwrite_output_dir = True,
    num_train_epochs = 5,
    per_gpu_train_batch_size = 8,
    save_steps = 10000,
    save_total_limit = 2,
)

trainer = transformers.Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = dataset,
    prediction_loss_only = True,
)
print('In training stage')
# Training
trainer.train()
# Sacing model
trainer.save_model("models/COVID_RoBERTa")
