# Covid-19 Misinformation Detection

This repository contains the code and data for Covid-19 Misinformation detection project.

Create the conda environment with the requirements "conda env create -f tfrs_env.yml"          

bert_pytorch.py - Code for using BERT based embeddings for downstream task.          

finetuning.py - Code for fine-tuning all models on all datasets.

intermediate_training/ - Contains the code to train models on CORD-19 with a pre-training objective (e.g, Masked Language Modeling)

supplemental_code/ - Code for creating plots for the paper and significance analysis.

baselines/ - Code for testing bi-LSTM baselines against transformer language models