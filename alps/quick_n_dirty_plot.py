import seaborn as sns
import pandas as pd

def bar_plot(csv_file):
    """"
Name","hex","F1"
"Relevant/Not Relevant","#3353b7","0.9327873265392336"
    """
    df = pd.read_csv(csv_file)
    sns.set_theme(style="whitegrid")
    colors = df["hex"].tolist()
    ax = sns.barplot(x="Category", y="F1-Score", data=df, palette=colors)
    # x labels rotation, without being cut off
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    
    # add values above the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    # set width and height of the figure
    ax.figure.set_size_inches(10, 6)
    # make sure the labels are not cut off
    ax.figure.tight_layout()
    # add custom color legend, according to colors and column "Group"
    colors_unique = df["hex"].unique()
    for i, group in enumerate(df["Group"].unique()):      
        ax.bar(0, 0, color=colors_unique[i], label=group)
    # add figure legend, with full white backgro0und
    ax.legend(loc="lower left", bbox_to_anchor=(0.1, 0.05), facecolor='white')
    ax.figure.tight_layout()
    ax.figure.savefig("barplot.png")
    return ax

if __name__ == "__main__":
    bar_plot("/home/vera/Documents/Arbeit/CRS/PsychNER/wandb_export_2024-09-30T21_37_37.567+02_00.csv")