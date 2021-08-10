grid run \
  --instance_type g4dn.xlarge finetuning.py \
  --dataset_path grid:covid-fakenews:2 \
  --model_name_or_path allenai/longformer-base-4096 \
  --dataset_name "['cmu', 'par', 'rec', 'fn19', 'coaid']"
  # --model_name_or_path "['allenai/scibert_scivocab_uncased', 'emilyalsentzer/Bio_ClinicalBERT', 'facebook/bart-base', 'deepset/covid_bert_base', 'vinai/bertweet-base', 'digitalepidemiologylab/covid-twitter-bert-v2', 'manueltonneau/clinicalcovid-bert-base-cased', 'manueltonneau/biocovid-bert-large-cased', 'allenai/longformer-base-4096']" \