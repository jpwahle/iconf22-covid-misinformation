# usage: python calc_scores.py path/to/run_folder/
import sys
import os
import numpy as np
import pandas as pd


def read_df(path):
    df = pd.read_csv(path)
    return df


def calc_mean_std(dfs):

    df_concat = pd.concat(dfs)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    df_stds = by_row_index.std()

    df_means["std"] = df_stds["eval/f1-macro"]
    df_means["dataset"] = dfs[0]["dataset_name"]
    df_means["model"] = dfs[0]["_name_or_path"]

    df_means = df_means.sort_values(by=["model"])

    df_means = df_means.round(2)
    df_means["mean_plusminus_std"] = df_means["eval/f1-macro"].astype(str) + "$\pm$" + df_means["std"].astype(str)

    df_means = df_means.pivot(index="model", columns="dataset", values="mean_plusminus_std")

    return df_means


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError("Please provide a path to the run folder.")

    filepath = sys.argv[1]

    dfs = []
    for filename in os.listdir(filepath):
        if filename.endswith(".csv"):
            df = read_df(os.path.join(filepath, filename))
            dfs.append(df)

    assert len(dfs) > 0, "No dataframes found in {}".format(filepath)

    df_mean_std = calc_mean_std(dfs)

    cols = ['cmu', 'coaid', 'rec', 'fn19', 'par']
    df_mean_std = df_mean_std[cols]

    print(df_mean_std)
    print(df_mean_std.to_latex(escape=False))
