import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--from_checkpoint', type=str, default=None)
    parser.add_argument('--max_sequence_length', type=int, default=128)
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--train_path', type=str, default='./datasets/CMU_MisCov_train.csv')
    parser.add_argument('--val_path', type=str, default='./datasets/CMU_MisCov_train.csv')
    parser.add_argument('--dataset_name', type=str, default='cmu')
    parser.add_argument('--train', action="store_true", default=False)
    parser.add_argument('--eval', action="store_true", default=False)
    parser.add_argument('--export_significance', action="store_true", default=False)
    parser.add_argument('--num_significance_samples', type=int, default=1000)
    parser.add_argument('--sample_size', type=float, default=0.01)
    parser.add_argument('--significance_output_path', type=str, default="./significance_results")


    # This argument will use the  above args and otherwise the defuault config for the respective model
    parser.add_argument('--use_standard_config', action="store_true", default=False)

    # Otherwise, these arguments will be used
    parser.add_argument('--epochs', type=int, default=3.0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    args = parser.parse_args()
    return args