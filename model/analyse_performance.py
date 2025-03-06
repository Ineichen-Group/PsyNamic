import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from confidenceinterval.bootstrap import bootstrap_ci
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve)
import math


EXPERIMENTS_PATH = "/home/vebern/scratch/PsyNamic/model/experiments"
DATE_PREFIX = "202502"  # Match folders with this prefix
TASKS = [
    "Data Collection", "Data Type", "Number of Participants", "Age of Participants", "Application Form",
    "Clinical Trial Phase", "Condition", "Outcomes", "Regimen", "Setting", "Study Control", "Study Purpose",
    "Substance Naivety", "Substances", "Sex of Participants", "Study Conclusion", "Study Type",
]

def get_metrics_ci(test_pred_file: str, threshold: float = 0.5, is_multilabel: bool = True) -> dict:
    """Computes evaluation metrics with confidence intervals."""
    print(f"Loading predictions from {test_pred_file}...")
    pred_df = pd.read_csv(test_pred_file, encoding="utf-8")
    pred_df["probability"] = pred_df["probability"].apply(
        lambda x: np.array(eval(x)))
    if is_multilabel:
        pred_df["label"] = pred_df["label"].apply(lambda x: np.array(eval(x)))
    else:
        pred_df["label"] = pred_df["label"].apply(lambda x: np.array(x))

    predictions = np.stack(pred_df["probability"].values)
    y_true = np.stack(pred_df["label"].values)

    # Convert probabilities to binary predictions
    if is_multilabel:
        y_pred = (predictions >= threshold).astype(int)
    else:
        y_pred = np.argmax(predictions, axis=1)

    # Compute metrics and confidence intervals
    print("Computing F1 score and confidence interval...")
    f1_score, f1_ci = bootstrap(custom_f1, y_true, y_pred)
    print("Computing accuracy and confidence interval...")
    accuracy, acc_ci = bootstrap(custom_accuracy, y_true, y_pred)
    print("Computing precision and confidence interval...")
    precision, prec_ci = bootstrap(custom_precision, y_true, y_pred)
    print("Computing recall and confidence interval...")
    recall, recall_ci = bootstrap(custom_recall, y_true, y_pred)

    metric_dict = {
        "f1": (f1_score, f1_ci),
        "accuracy": (accuracy, acc_ci),
        "precision": (precision, prec_ci),
        "recall": (recall, recall_ci),
    }
    return metric_dict


def bootstrap(metric, y_true, y_pred) -> tuple:
    """Computes bootstrap confidence intervals."""
    print(f"Running bootstrap for {metric.__name__}...")
    score, ci = bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric=metric,
        confidence_level=0.95,
        n_resamples=9999,
        method="bootstrap_bca",
        random_state=42,
    )
    return score, ci


def custom_f1(true_labels, pred_labels):
    return f1_score(true_labels, pred_labels, average="weighted", zero_division=0)


def custom_accuracy(true_labels, pred_labels):
    return accuracy_score(true_labels, pred_labels)


def custom_precision(true_labels, pred_labels):
    return precision_score(true_labels, pred_labels, average="weighted", zero_division=0)


def custom_recall(true_labels, pred_labels):
    return recall_score(true_labels, pred_labels, average="weighted", zero_division=0)


def plot_precision_recall_curve(test_pred_file: str, best_model: str, ax: plt.Axes):
    """Plots Precision-Recall curve for the best model at varying thresholds."""
    print(f"Plotting Precision-Recall curve for {best_model}...")

    # Load predictions and true labels
    pred_df = pd.read_csv(test_pred_file, encoding="utf-8")
    pred_df["probability"] = pred_df["probability"].apply(
        lambda x: np.array(eval(x)))
    pred_df["label"] = pred_df["label"].apply(lambda x: np.array(eval(x)))

    predictions = np.stack(pred_df["probability"].values)
    y_true = np.stack(pred_df["label"].values)

    # Calculate precision, recall, and thresholds for various thresholds
    precisions, recalls, thresholds = precision_recall_curve(
        y_true.ravel(), predictions.ravel())

    # Calculate F1 score at each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero

    # Plot Precision-Recall curve
    ax.plot(recalls, precisions, label=f'{best_model} Precision-Recall Curve')

    # Plot F1 score curve
    ax.plot(recalls, f1_scores,
            label=f'{best_model} F1 Score', linestyle="--", color="green")

    # Set labels and title
    ax.set_xlabel("Recall")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision-Recall & F1 Curve for {best_model}")

    # Add a legend
    ax.legend(loc="lower left")


def plot_metric_comparison(df, metric, metric_ci_lower, metric_ci_upper, save_path):
    """General function to plot a comparison for each metric."""
    # Sort by the chosen metric
    df_sorted = df.sort_values(by=metric, ascending=False)

    # Set up the color palette
    model_colors = {
        "pubmedbert": "#1f77b4", "biomedbert-abstract": "#ff7f0e", "scibert": "#2ca02c",
        "biobert": "#d62728", "clinicalbert": "#9467bd", "biolinkbert": "#8c564b"
    }

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Model", y=metric, data=df_sorted,
                     hue="Model", dodge=False, palette=model_colors)

    # Add confidence intervals as error bars
    for index, row in df_sorted.iterrows():
        yerr_lower = row[metric] - row[metric_ci_lower]  # Lower CI
        yerr_upper = row[metric_ci_upper] - row[metric]  # Upper CI
        ax.errorbar(index, row[metric],
                    yerr=[[yerr_lower], [yerr_upper]],
                    fmt='none', color='black', capsize=5)
        ax.text(index, row[metric_ci_lower] - 0.04,
                f"{row[metric_ci_lower]:.3f}", ha="center", va="bottom", color="black", fontsize=8)
        ax.text(index, row[metric_ci_upper] + 0.02,
                f"{row[metric_ci_upper]:.3f}", ha="center", va="bottom", color="black", fontsize=8)

    # Annotate the values on top of the bars
    for index, row in df_sorted.iterrows():
        ax.text(index, row[metric] - 0.2,  # Place it roughly in the middle of the bar
                f"{row[metric]:.3f}", ha="center", color="black")

    # Labeling the plot
    ax.set_xlabel("Model")
    ax.set_ylabel(f"{metric}")
    ax.set_title(f"Model Performance Comparison Based on {metric}")

    # Fix the x-ticks (Set them manually to avoid warning)
    ax.set_xticks(np.arange(len(df_sorted)))
    ax.set_xticklabels(df_sorted['Model'], rotation=45, ha="right")

    # Highlight the best model
    best_model = df_sorted.iloc[0]["Model"]
    ax.set_title(
        f"Best Model: {best_model} ({metric}: {df_sorted.iloc[0][metric]:.3f})", fontsize=14)

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def make_model_comparison_plot():
    """Creates multi-task, multi-metric comparison plots."""
    print("Finding experiment directories...")
    task_model_performance = {}

    print(f"Identifying tasks: {TASKS}")
    tasks = [task.lower().replace(' ', '_') for task in TASKS]

    for task in tasks:
        print(f"Processing task: {task}")
        model_performance = []
        outfile = f"model_performance_{task}.csv"
        outfile = os.path.join(os.path.dirname(EXPERIMENTS_PATH), outfile)
        if os.path.exists(outfile):
            model_performance = pd.read_csv(outfile)
        else:
            model_performance = pd.DataFrame(get_metrics_from_prediction(task))
            model_performance.to_csv(outfile, index=False)

        task_model_performance[task] = model_performance

    # Generate a multi-metric multi-task plot
    plot_multi_task_comparison(task_model_performance,
                               metrics=["F1", "Accuracy",
                                        "Precision", "Recall"],
                               save_dir="experiments/performance_plots")


def get_metrics_from_prediction(task: str) -> list[dict]:
    model_performance = []
    for exp_dir in os.listdir(EXPERIMENTS_PATH):
        if task not in exp_dir:
            continue  # Skip directories that don't match this task

        exp_path = os.path.join(EXPERIMENTS_PATH, exp_dir)
        test_pred_file = os.path.join(exp_path, "test_predictions.csv")
        params_file = os.path.join(exp_path, "params.json")

        if not (os.path.exists(test_pred_file) and os.path.exists(params_file)):
            print(f"Skipping {exp_dir} due to missing files.")
            continue

        # Read params.json to check if it's multilabel
        with open(params_file, "r", encoding="utf-8") as f:
            params = json.load(f)
            is_multilabel = params.get("is_multilabel", True)

        # Compute evaluation metrics
        print(f"Computing metrics for {exp_dir}...")
        metrics = get_metrics_ci(
            test_pred_file, is_multilabel=is_multilabel)
        model = exp_dir.split("_")[0]  # Extract model name
        model_performance.append({
            "Model": model,
            "F1": metrics["f1"][0],
            "F1 CI Lower": metrics["f1"][1][0],
            "F1 CI Upper": metrics["f1"][1][1],
            "Accuracy": metrics["accuracy"][0],
            "Accuracy CI Lower": metrics["accuracy"][1][0],
            "Accuracy CI Upper": metrics["accuracy"][1][1],
            "Precision": metrics["precision"][0],
            "Precision CI Lower": metrics["precision"][1][0],
            "Precision CI Upper": metrics["precision"][1][1],
            "Recall": metrics["recall"][0],
            "Recall CI Lower": metrics["recall"][1][0],
            "Recall CI Upper": metrics["recall"][1][1],
        })

    return model_performance


def plot_all_metrics(csv_file: str, outdir: str, task: str):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    task = task.lower().replace(' ', '_')

    # Ensure the required columns are present
    required_columns = [
        'Model', 'F1 Score', 'F1 CI Lower', 'F1 CI Upper',
        'Accuracy', 'Accuracy CI Lower', 'Accuracy CI Upper',
        'Precision', 'Precision CI Lower', 'Precision CI Upper',
        'Recall', 'Recall CI Lower', 'Recall CI Upper'
    ]

    if not all(col in df.columns for col in required_columns):
        print("Missing required columns in the CSV.")
        return

    # Plot for F1 Score
    plot_metric_comparison(
        df, 'F1 Score', 'F1 CI Lower', 'F1 CI Upper', os.path.join(
            outdir, f'f1_score_comparison_{task}.png')
    )

    # Plot for Accuracy
    plot_metric_comparison(
        df, 'Accuracy', 'Accuracy CI Lower', 'Accuracy CI Upper', os.path.join(
            outdir, f'accuracy_comparison_{task}.png')
    )

    # Plot for Precision
    plot_metric_comparison(
        df, 'Precision', 'Precision CI Lower', 'Precision CI Upper', os.path.join(
            outdir, f'precision_comparison_{task}.png')
    )

    # Plot for Recall
    plot_metric_comparison(
        df, 'Recall', 'Recall CI Lower', 'Recall CI Upper', os.path.join(
            outdir, f'recall_comparison_{task}.png')
    )


def plot_multi_task_comparison(task_model_performance, metrics, save_dir):
    """
    Generates a separate multi-panel figure for each metric, where each subplot represents a task
    and displays model performances with proper error bars.

    Parameters:
        task_model_performance (dict): Dictionary where keys are task names and values are performance DataFrames.
        metrics (list): List of metric names to plot (e.g., ["F1 Score", "Accuracy"]).
        save_dir (str): Directory path to save the final multi-plot figures.
    """
    os.makedirs(save_dir, exist_ok=True)

    model_colors = {
        "pubmedbert": "#1f77b4", "biomedbert-abstract": "#ff7f0e", "scibert": "#2ca02c",
        "biobert": "#d62728", "clinicalbert": "#9467bd", "biolinkbert": "#8c564b"
    }

    num_tasks = len(task_model_performance)

    for metric in metrics:
        ncols = 3  # 3 plots per row
        nrows = math.ceil(num_tasks / ncols)  # Number of rows needed

        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows), sharex=False)

        # Flatten the axes to make sure they are iterable in case of multiple rows and columns
        axes = axes.flatten()

        for i, (task, model_performance) in enumerate(task_model_performance.items()):
            df_sorted = pd.DataFrame(model_performance).sort_values(by=metric, ascending=False).reset_index(drop=True)
            ax = axes[i]
            ax.set_title(f"{task} - {metric}")
            ax.set_xticks(np.arange(len(df_sorted)))
            ax.set_xticklabels(df_sorted["Model"], rotation=45, ha="right")
            ax.set_ylabel(metric)
            ax.set_xlabel("Model")
            ax.set_ylim(0, 1)

            # Create bar plot with proper hue setting
            sns.barplot(x="Model", y=metric, hue="Model", data=df_sorted, ax=ax,
                        palette=model_colors, legend=False, errorbar=None)

            for index, row in df_sorted.iterrows():
                ax.text(index, row[metric] - 0.2,  # Place it roughly in the middle of the bar
                        f"{row[metric]:.3f}", ha="center", color="black")
                
                # Get CI column names
                ci_lower_col = f"{metric} CI Lower"
                ci_upper_col = f"{metric} CI Upper"

                # Set error bars
                yerr_lower = row[metric] - row[ci_lower_col]
                yerr_upper = row[ci_upper_col] - row[metric]
                ax.errorbar(index, row[metric],
                            yerr=[[yerr_lower], [yerr_upper]],
                            fmt='none', color='black', capsize=5)
                ax.text(index, row[ci_lower_col] - 0.04,
                        f"{row[ci_lower_col]:.3f}", ha="center", va="bottom", color="black", fontsize=8)
                ax.text(index, row[ci_upper_col] + 0.02,
                        f"{row[ci_upper_col]:.3f}", ha="center", va="bottom", color="black", fontsize=8)
        for j in range(num_tasks, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{metric}_comparison.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {save_path}")


def main():
    print("Starting model comparison plot generation...")
    make_model_comparison_plot()
    print("Process completed.")

    # file = '/home/vebern/scratch/PsyNamic/model/model_performance_substances.csv'
    # plot_all_metrics(file, 'experiments/performance_plots', 'Substances')


if __name__ == "__main__":
    main()
