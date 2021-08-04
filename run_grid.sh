grid run \
  --datastore_name trafficking-10k \     
  --datastore_version 2 \
  --datastore_mount_dir /trafficking-10k \
  --instance_type g4dn.xlarge \
  finetuning.py \
  --model_name_or_path "[bert-base-uncased, distilbert-base-uncased, roberta-base, allenai/scibert_scivocab_uncased, emilyalsentzer/Bio_ClinicalBERT, lordtt13/COVID-SciBERT, facebook/bart-base, covid-roberta, covid-bart, covid-deberta, deepset/covid_bert_base, vinai/bertweet-base, digitalepidemiologylab/covid-twitter-bert-v2, manueltonneau/clinicalcovid-bert-base-cased, manueltonneau/biocovid-bert-large-cased, covid-longformer, covid-twitter-bert, allenai/longformer-base-4096]" \
  --export_significance \
  --train \
  --eval \
  --dataset_name "[cmu, par, rec, fn19, coaid]" \
  --dataset_path ~/covid-fakenews


python3 finetuning.py \
  --model_name_or_path bert-base-uncased \
  --export_significance \
  --train \
  --eval \
  --dataset_name cmu \
  --dataset_path ~