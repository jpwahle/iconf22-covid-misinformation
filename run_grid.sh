grid run \
  --instance_type g4dn.xlarge \
  --use_spot finetuning.py \
  --dataset_path grid:covid-fakenews:2 \
  --model_name_or_path "['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base']" \
  --dataset_name "['cmu', 'par', 'rec', 'fn19', 'coaid']"