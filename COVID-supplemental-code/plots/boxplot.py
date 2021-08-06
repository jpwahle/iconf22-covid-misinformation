import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def box_plot(perspective="dataset", show=False):
    f = plt.figure(figsize=(7, 5))
    df = pd.read_excel("results_combined.xlsx", sheet_name="fine_tune_plots", nrows=17)
    df = pd.melt(df, id_vars=['Models'], value_vars=['CMU-MisCov19', 'CoAID', 'ReCOVery', 'COVID19FN', 'COVID-CQ'])
    if perspective == "model":
        ax = sns.boxplot(x='Models', y='value', data=df)
    elif perspective == "dataset":
        ax = sns.boxplot(x='variable', y='value', data=df, width=0.4)
    ax.set_xlabel('')
    ax.set_ylabel('F1-Macro')

    plt.xticks(rotation=40, ha="right")
    f.savefig("boxplot{}.pdf".format(perspective), bbox_inches='tight')
    if show:
        plt.show()

def line_plot(show=False):
    f = plt.figure(figsize=(7, 5))

    df = pd.read_excel("results_combined.xlsx", sheet_name="fine_tune_plots", nrows=17)
    df = pd.melt(df, id_vars=['Models'], value_vars=['CMU-MisCov19', 'CoAID', 'ReCOVery', 'COVID19FN', 'COVID-CQ'])
    df['Dataset'] = df['variable']
    ax = sns.lineplot(data=df, x="Models", y="value", hue="Dataset", marker="o", sort=False)
    ax.set_xlabel('')
    ax.set_ylabel('F1-Macro')
    
    plt.xticks(rotation=40, ha="right")
    f.savefig("lineplot.pdf", bbox_inches='tight')
    if show:
        plt.show()

def bar_plot(show=False):
    f = plt.figure(figsize=(7, 5))

    df = pd.read_excel("results_combined.xlsx", sheet_name="fine_tune_plots", nrows=17)
    df = pd.melt(df, id_vars=['Models'], value_vars=['CMU-MisCov19', 'CoAID', 'ReCOVery', 'COVID19FN', 'COVID-CQ'])
    ax = sns.barplot(x="variable", y="value", hue="Models", palette="mako", data=df)
    ax.set_xlabel('')
    f.savefig("barplot.pdf", bbox_inches='tight')
    ax.set_ylabel('F1-Macro')

    if show:
        plt.show()
        
if __name__ == '__main__':
    # box_plot(perspective="model")
    # box_plot(perspective="dataset")
    line_plot()
    # bar_plot()