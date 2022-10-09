# Covid-19 Misinformation Detection

[![arXiv](https://img.shields.io/badge/arXiv-2111.07819-b31b1b.svg)](https://arxiv.org/abs/2111.07819)

This repository contains the code and data for the paper ["Testing the Generalization of Neural Language Models for COVID-19 Misinformation Detection"](https://arxiv.org/abs/2111.07819).

## How to use

We are providing two ways to setup this project.

<details> <summary> Getting Started </summary>
  <br/>
  First install conda from <a href="https://docs.conda.io/en/latest/" >here</a>.
  
  Next, create the conda environment with the requirements "conda env create -f tfrs_env.yml"          
</details>

<details> <summary> Training and testing </summary>
<br/>
The following files and folders contain the code to reproduce the experiments from our paper:
  
- bert_pytorch.py - Code for using BERT based embeddings for downstream task.          


- finetuning.py - Code for fine-tuning all models on all datasets.


- intermediate_training/ - Contains the code to train models on CORD-19 with a pre-training objective (e.g, Masked Language Modeling)


- supplemental_code/ - Code for creating plots for the paper and significance analysis.


- baselines/ - Code for testing bi-LSTM baselines against transformer language models
</details>




## How to cite
```tex
@inproceedings{Wahle2022a,
  title        = {{Testing} the {Generalization} of {Neural} {Language} {Models} for {COVID}-19 {Misinformation} {Detection}},
  author       = {Wahle, Jan Philip and Ashok, Nischal and Ruas, Terry and Meuschke, Norman and Ghosal, Tirthankar and Gipp, Bela},
  year         = 2022,
  month        = {February},
  booktitle    = {Proceedings of the iConference},
  location     = {Virtual Event},
}
```
