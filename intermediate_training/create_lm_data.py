import pandas as pd
import re
from pathlib import Path
import os

# Basic Pre-Processing
def basicPreprocess(text):
    processed_text = text.lower()
    processed_text = re.sub(r'\W +', ' ', processed_text)
    return processed_text

print("Reading the data")
complete_df = pd.read_csv("data/clean_df.csv")
data = complete_df.sample(frac = 1).sample(frac = 1)
data.dropna(inplace = True)
print("Processing the data")
data = data["text"].apply(basicPreprocess).replace("\n"," ")

lm_data_dir = "lm_data_text"
train_split = 0.9
train_data_size = int(len(data)*train_split)

print("Writing the data")
with open(os.path.join(lm_data_dir,'train.txt') , 'w') as f:
    for item in data[:train_data_size].tolist():
        f.write("%s\n" % item)

with open(os.path.join(lm_data_dir,'eval.txt') , 'w') as f:
    for item in data[train_data_size:].tolist():
        f.write("%s\n" % item)