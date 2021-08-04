import os
import json
import torch
import random
import numpy as np
from pathlib import Path
from args import parse_args
from data import COVIDDataset
from typing import List, Dict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, EvalPrediction


def create_compute_metrics_fn(average="macro"):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=average)
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1-macro': f1,
            'precision': precision,
            'recall': recall
        }
    return compute_metrics

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def generate_tokenize_fn(tokenizer):
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    return tokenize_fn

def init_model(model_name: str, num_classes: int) -> AutoModelForSequenceClassification:
    config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    return model

def init_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)

def load_model_list(path: str) -> List[Dict[str, str]]:
    with open(path, "r") as f:
        data = json.load(f)
    return data['models']

def load_weights_from_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))

def main():

    # TODO: Depending on the model size, we may need to reduce the batch size and sequence length
    # model_list = load_model_list(args.model_list_path)


    # Number of classes per dataset (hardcoded) just for testing the num_labels function that automatically calculates the number of labels
    num_class_dict = {
        "cmu": 16,
        "coaid": 3,
        "fn19": 3,
        "par": 3,
        "rec": 3
    }
    
    # Parse arguments
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.random_seed)

    # Init tokenizer
    tokenizer = init_tokenizer(args.model_name_or_path)

    # Create Datasets
    train_dataset = COVIDDataset(data_path=args.train_path, tokenizer=tokenizer, dataset_name=args.dataset_name)
    test_dataset = COVIDDataset(data_path=args.val_path, tokenizer=tokenizer, dataset_name=args.dataset_name)

    num_labels = train_dataset.num_labels()
    assert num_labels == num_class_dict.get(args.dataset_name), "Number of labels in train dataset does not align with expected number of labels for {}".format(args.dataset_name)

    # Load model (and checkpoint)
    model = init_model(args.model_name_or_path, num_class_dict[args.dataset_name])
    if args.from_checkpoint:
        load_weights_from_checkpoint(model, args.from_checkpoint)

    # Configure Training Arguments
    if args.use_standard_config:
        training_args = TrainingArguments("default_trainer")
    else:
        training_args = TrainingArguments(
            "param_trainer",
            num_train_epochs=args.epochs,
            adam_epsilon=args.adam_epsilon,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.valid_batch_size,
            warmup_steps=args.warmup_steps,
        )

    # Compile Trainer
    compute_metric = create_compute_metrics_fn("macro")
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metric
    )

    # Train
    if args.train:
        trainer.train()

    # Eval
    if args.eval:
        trainer.evaluate()

    if args.export_significance:
        # Generate predictions
        preds, labels, _ = trainer.predict(test_dataset=test_dataset)

        # The size of samples
        num_samples = int(len(labels) * args.sample_size)

        # Create dirs if not exist
        Path(os.path.join(args.significance_output_path, args.dataset_name)).mkdir(parents=True, exist_ok=True)

        # Write to txt file
        with open(os.path.join(args.significance_output_path, args.dataset_name, args.model_name_or_path + '.txt'), "w") as f:

            # For each sample
            for i in range(args.num_significance_samples):

                # Draw randomly the num_samples
                index_list = [i for i in range(len(labels))]
                chosen_indices = random.sample(index_list, num_samples)

                # Calculate F1
                score = compute_metric(EvalPrediction(predictions=preds[chosen_indices], label_ids=labels[chosen_indices]))

                # Write to file
                f.write(str(score["f1-macro"]) + os.linesep)

if __name__ == '__main__':
    main()