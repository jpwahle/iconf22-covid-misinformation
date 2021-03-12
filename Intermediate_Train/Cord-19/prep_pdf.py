# All imports
import os
import re
import json
import utils
import random
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm

warnings.filterwarnings('ignore')

# Loading data
pdf_json = 'document_parses/pdf_json/'
filenames = os.listdir(pdf_json)
print("Number of articles retrieved from pdf_json articles:", len(filenames))

# Cleaning the files
cleaned_files = []

for file in tqdm(os.listdir(pdf_json)):
    filename = pdf_json + file
    file = json.load(open(filename, 'rb'))
    features = [
        file['paper_id'],
        file['metadata']['title'],
        utils.format_authors(file['metadata']['authors']),
        utils.format_authors(file['metadata']['authors'], 
                       with_affiliation = True),
        utils.format_body(file['abstract']),
        utils.format_body(file['body_text']),
        utils.format_bib(file['bib_entries']),
        file['metadata']['authors'],
        file['bib_entries']
    ]
    
    cleaned_files.append(features)
    
    del file
    del features

# Converting to csv

col_names = [
    'paper_id', 
    'title', 
    'authors',
    'affiliations', 
    'abstract', 
    'text', 
    'bibliography',
    'raw_authors',
    'raw_bibliography'
]

clean_df = pd.DataFrame(cleaned_files, columns = col_names)
del cleaned_files
print(clean_df.head(2))

clean_df.isna().sum()
clean_df.to_csv("data/clean_df.csv")

