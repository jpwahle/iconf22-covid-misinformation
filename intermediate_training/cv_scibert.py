import os
import re
import transformers
import torch

import pandas as pd

from collections import Counter

################ Setup ##############
#####################################
# If there's a GPU available...
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

def basicPreprocess(text):
    processed_text = text.lower()
    processed_text = re.sub(r"[^a-zA-Z0-9]+", ' ', processed_text)
    return processed_text

complete_df = pd.read_csv("data/clean_df.csv")
data = complete_df.sample(frac = 1).sample(frac = 1)
data.dropna(inplace = True)
del complete_df
data = data["abstract"].apply(basicPreprocess).replace("\n"," ")
text = ''
for i in data.values:
    text += i
del data
counter = Counter(text.split())
del text
vocab = []
for keys, values in counter.items():
    if(values > 100 and values < 10000):
        vocab.append(keys)
print(len(vocab))

# Loading pre-trained model
tokenizer = transformers.AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = transformers.AutoModelWithLMHead.from_pretrained('allenai/scibert_scivocab_uncased').to('cuda')
print(model.config)
# # Code for appending vocab
# print(len(tokenizer))
# tokenizer.add_tokens(vocab)
# print(len(tokenizer))
# model.resize_token_embeddings(len(tokenizer)) 
# model.config
# del vocab

# # Fine tuning
# try:
#     os.mkdir('models/COVID-scibert-latest')
#     tokenizer.save_pretrained('models/COVID-scibert-latest')
# except FileExistsError:
#     print('File already exists')
    
print('Proceeding with the dataset')
dataset = transformers.LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = "lm_data/train.txt",
    block_size = 128,
)
data_collator = transformers.DataCollatorForLanguageModeling(
    tokenizer = tokenizer, mlm = True, mlm_probability = 0.15
)

print("Data Collator stage complete")

training_args = transformers.TrainingArguments(
    output_dir = "models/COVID-scibert-latest",
    overwrite_output_dir = True,
    num_train_epochs = 5,
    per_device_train_batch_size = 32,
    save_steps = 10000,
    save_total_limit = 3,
)

trainer = transformers.Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = dataset,
    prediction_loss_only = True,
)
print("Training args initialized")
print('In training stage')
trainer.train()
trainer.save_model("models/COVID-scibert-latest")

