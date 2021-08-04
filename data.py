import torch
import pandas as pd
from typing import Tuple
from torch.utils.data import Dataset


def text_label(df):
    return list(df['text']), list(df['label'])

def title_content_label(df):
    return list(df['title']), list(df['content']), list(df['label'])

def premise_text_stance(df):
    return list(df['premise']), list(df['text']), list(df['stance'])


class COVIDDataset(Dataset):
    def __init__(self, data_path, tokenizer, dataset_name):

        self.data_path = data_path
        self.tokenizer = tokenizer    
        self.dataset_name = dataset_name

        # Each dataset has a different data loading function
        self.dataset_dict = {
            "cmu": text_label,
            "coaid": title_content_label,
            "fn19": title_content_label,
            "rec": title_content_label,
            "par": premise_text_stance,
        }
        
        # Get tokenized text according to the dataset and labels
        self.encodings, self.labels = self.load_data()


    def load_data(self) -> Tuple[dict, torch.tensor]:
        data_df = pd.read_csv(self.data_path)

        # CMU has one text and needs to be tokenzied [CLS] text [SEP] (depending on the model the CLS and SEP change automatically)
        if self.dataset_name == "cmu":
            X, y = self.dataset_dict[self.dataset_name](data_df)
            X_encodings = self.tokenizer(X, truncation=True, padding=True)
        # Everything else has two texts and needs [CLS] text1 [SEP] text2 [SEP] (depending on the model the CLS and SEP change automatically)
        else:
            X_1, X_2, y = self.dataset_dict[self.dataset_name](data_df)
            X_encodings = self.tokenizer(X_1, X_2, truncation=True, padding=True)

        return (X_encodings, y)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def num_labels(self):
        return max(self.labels) - min(self.labels) + 1

    
class CMUDataset(COVIDDataset):
    def __init__(self, data_path, tokenizer):
        super(CMUDataset, self).__init__(data_path, tokenizer)

