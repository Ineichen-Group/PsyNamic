import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from confidenceinterval.bootstrap import bootstrap_ci
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve)


EXPERIMENTS_PATH = "/home/vebern/scratch/PsyNamic/model/experiments"
DATE_PREFIX = "202502"  # Match folders with this prefix
# TASKS = [
#         "Data Collection", "Data Type", "Number of Participants", "Age of Participants", "Application Form",
#         "Clinical Trial Phase", "Condition", "Outcomes", "Regimen", "Setting", "Study Control", "Study Purpose",
#         "Substance Naivety", "Substances", "Sex of Participants", "Study Conclusion", "Study Type",
#     ]
TASKS = ["Substances"]


def get_metrics_ci(test_pred_file: str, threshold: float = 0.5, is_multilabel: bool = True) -> dict:
    """Computes evaluation metrics with confidence intervals."""
    print(f"Loading predictions from {test_pred_file}...")
    pred_df = pd.read_csv(test_pred_file, encoding="utf-8")
    pred_df["probability"] = pred_df["probability"].apply(
        lambda x: np.array(eval(x)))
    pred_df["label"] = pred_df["label"].apply(lambda x: np.array(eval(x)))

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
    pred_df = pd.read_csv(test_pred_file, encoding="utf-8")
    pred_df["probability"] = pred_df["probability"].apply(
        lambda x: np.array(eval(x)))
    pred_df["label"] = pred_df["label"].apply(lambda x: np.array(eval(x)))

    predictions = np.stack(pred_df["probability"].values)
    y_true = np.stack(pred_df["label"].values)

    # Calculate precision and recall for various thresholds
    precisions, recalls, thresholds = precision_recall_curve(
        y_true.ravel(), predictions.ravel())

    # Plot Precision-Recall curve
    ax.plot(recalls, precisions, label=f'{best_model}')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve for {best_model}")
    ax.legend(loc="lower left")


def make_model_comparison_plot():
    """Creates evaluation metric plots for each task."""
    print("Finding experiment directories...")
    task_model_performance = {}

    print(f"Identifying tasks: {TASKS}")
    tasks = [task.lower().replace(' ', '_') for task in TASKS]

    for task in tasks:
        print(f"Processing task: {task}")
        model_performance = []

        for exp_dir in os.listdir(EXPERIMENTS_PATH):
            if task not in exp_dir:
                continue  # Skip directories that don't match this task

            exp_path = os.path.join(EXPERIMENTS_PATH, exp_dir)
            test_pred_file = os.path.join(exp_path, "test_predictions.csv")
            params_file = os.path.join(exp_path, "params.json")

            if not os.path.exists(test_pred_file) or not os.path.exists(params_file):
                print(f"Skipping {exp_dir} due to missing files.")
                continue

            # Read params.json to check if it's multilabel
            with open(params_file, "r", encoding="utf-8") as f:
                params = json.load(f)
                is_multilabel = params.get("is_multilabel", True)
                if not is_multilabel:
                    continue

            # Compute evaluation metrics
            print(f"Computing metrics for {exp_dir}...")
            metrics = get_metrics_ci(
                test_pred_file, is_multilabel=is_multilabel)
            model = exp_dir.split("_")[0]  # Extract model name
            model_performance.append({
                "Model": model,
                "F1 Score": metrics["f1"][0],
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

        if model_performance:
            task_model_performance[task] = model_performance

            # Save performance to CSV
            task_df = pd.DataFrame(model_performance)
            task_df.to_csv(f"model_performance_{task}.csv", index=False)

            # Plot the Precision-Recall curve for the best model
            best_model = task_df.loc[task_df["F1 Score"].idxmax(), "Model"]
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_precision_recall_curve(test_pred_file, best_model, ax)
            plt.savefig(f"precision_recall_curve_{task}_{best_model}.png")
            plt.close()

    if not task_model_performance:
        print("No valid data found. Exiting.")
        return

    # Plot setup
    print(f"Generating plots for {len(task_model_performance)} tasks...")
    num_tasks = len(task_model_performance)
    num_columns = 3
    num_rows = (num_tasks // num_columns) + \
        (1 if num_tasks % num_columns != 0 else 0)
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 6 * num_rows))
    axes = axes.flatten()

    model_colors = {
        "pubmedbert": "#1f77b4", "biomedbert-abstract": "#ff7f0e", "scibert": "#2ca02c",
        "biobert": "#d62728", "clinicalbert": "#9467bd", "biolinkbert": "#8c564b"
    }

    for i, (task, model_performance) in enumerate(task_model_performance.items()):
        print(f"Plotting performance for task: {task}")
        task_data = pd.DataFrame(model_performance)

        max_model = task_data.loc[task_data["F1 Score"].idxmax(), "Model"]
        bar_colors = [model_colors.get(
            model, "grey") if model == max_model else "grey" for model in task_data["Model"]]

        sns.barplot(x="Model", y="F1 Score", data=task_data, ax=axes[i], palette=bar_colors,
                    ci=None)

        for index, row in task_data.iterrows():
            axes[i].text(index, row["F1 Score"] + 0.02, f"{row['F1 Score']:.3f}",
                         color="black", ha="center")
            axes[i].errorbar(index, row["F1 Score"], yerr=[[row["F1 CI Lower"]], [row["F1 CI Upper"]]],
                             fmt='none', color='black', capsize=5)

        axes[i].set_title(task)
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].set_ylabel("F1 Score")
        axes[i].set_xlabel("Model")
        axes[i].set_ylim(0, 1)

    # Hide unused subplots
    for j in range(num_tasks, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    print("Saving plot as 'model_performance_task_plots.png'...")
    plt.savefig("model_performance_task_plots.png", bbox_inches="tight")


def main():
    print("Starting model comparison plot generation...")
    make_model_comparison_plot()
    print("Process completed.")


if __name__ == "__main__":
    main()
