import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
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
    "Substance Naivety", "Substances", "Sex of Participants", "Study Conclusion", "Study Type", "Relevant"
]

MODEL_COLORS = {
    "pubmedbert": "#1f77b4", "biomedbert-abstract": "#ff7f0e", "scibert": "#2ca02c",
    "biobert": "#d62728", "clinicalbert": "#9467bd", "biolinkbert": "#8c564b"
}


def extract_predictions(test_pred_file: str, is_multilabel: bool, threshold: float = 0.5) -> tuple:
    """ Extracts labels, predicte labels and probability distribution from a test_predicitions.csv

        For multilabel classification, it uses the threshold to determin labels (in predictions.csv it's 0.5 per default)
    """
    pred_df = pd.read_csv(test_pred_file, encoding="utf-8")
    pred_df["probability"] = pred_df["probability"].apply(
        lambda x: np.array(eval(x)))
    if is_multilabel:
        pred_df["label"] = pred_df["label"].apply(lambda x: np.array(eval(x)))
        pred_df["prediction"] = pred_df["prediction"].apply(
            lambda x: np.array(eval(x)))
    else:
        pred_df["label"] = pred_df["label"].apply(lambda x: np.array(x))
        pred_df["prediction"] = pred_df["prediction"].apply(
            lambda x: np.array(x))

    y_pred = np.stack(pred_df["prediction"].values)
    probs = np.stack(pred_df["probability"].values)
    y_true = np.stack(pred_df["label"].values)

    if threshold != 0.5:
        if is_multilabel:
            y_pred = (probs >= threshold).astype(int)
        else:
            y_pred = np.argmax(probs, axis=1)

    return y_true, y_pred, probs


def get_metrics_ci(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes evaluation metrics with confidence intervals."""
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


def bootstrap(metric: callable, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
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


def precision_recall_curve(true_labels: np.ndarray, pred_probs: np.ndarray, is_multilabel: bool, task: str, ax=None, save_path=None, nr_thresholds=50, best_model: str = ''):
    """Produces a precision recal curve"""
    precisions = []
    recalls = []
    f1_scores = []

    thresholds = np.linspace(0, 1, nr_thresholds)
    for threshold in thresholds:
        if is_multilabel:
            pred_labels = (pred_probs >= threshold).astype(int)
        else:
            return

        precisions.append(custom_precision(true_labels, pred_labels))
        recalls.append(custom_recall(true_labels, pred_labels))
        f1_scores.append(custom_f1(true_labels, pred_labels))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(thresholds, precisions, label=f'Precision', color='blue')
    ax.plot(thresholds, recalls, label=f'Recall', color='orange')
    ax.plot(thresholds, f1_scores, label=f'F1 Score',
            linestyle='--', color='green')

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(
        f"Performance at Varying Thresholds for {task} by {best_model}")

    max_f1_idx = np.argmax(f1_scores)
    max_f1_threshold = thresholds[max_f1_idx]
    ax.axvline(x=max_f1_threshold, color='purple',
               linestyle=':', label='Max F1 Threshold')

    precision_at_max_f1 = precisions[max_f1_idx]

    acceptable_precision = recalls[max_f1_idx]
    acceptable_precision_i = max_f1_idx
    for i in range(max_f1_idx, 0, -1):
        if precisions[i] >= precision_at_max_f1 - 0.05:
            acceptable_precision = precisions[i]
            acceptable_precision_i = i
        else:
            break
    ax.axvline(x=thresholds[acceptable_precision_i], color='red',
               linestyle=':', label='Optimization for higher recall, -5%')

    ax.scatter([thresholds[acceptable_precision_i]], [
               acceptable_precision], color='red', zorder=3)
    ax.text(thresholds[acceptable_precision_i], acceptable_precision, f'Th={thresholds[acceptable_precision_i]:.2f}',
            verticalalignment='bottom', horizontalalignment='right', color='red', fontsize=10)

    ax.legend(loc="lower left")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return ax


def plot_precision_recall_curve_all_tasks(task_model_performance: dict, save_dir: str) -> dict:
    """Creates a multiplot of all precision recall plots for all multilabel cases"""
    ncols = 3
    nrows = math.ceil(len(task_model_performance) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows))
    axes = axes.flatten()
    best_models_task = {}

    for i, (task, _) in enumerate(task_model_performance.items()):
        ax = axes[i]
        best_models = task_model_performance[task][task_model_performance[task]
                                                   ['F1'] == task_model_performance[task]['F1'].max()].copy()

        if len(best_models) > 1:
            print(
                f"\nMultiple models have the highest F1 score for task: {task}")

            # Create a numbered list from 1 onwards
            model_choices = list(best_models.itertuples(index=False))
            for idx, row in enumerate(model_choices, start=1):
                print(f"{idx}: {row.Model} (F1: {row.F1:.4f})")

            while True:
                try:
                    choice = int(
                        input(f"Select a model for {task} (1-{len(model_choices)}): "))
                    if 1 <= choice <= len(model_choices):
                        best_model = model_choices[choice - 1].Model
                        break
                    else:
                        print("Invalid choice. Please enter a number from the list.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

        else:
            best_model = best_models.loc[best_models.index[0], 'Model']

        best_models_task[task] = best_model
        model_path = find_model_path(task, best_model)
        test_pred_file = os.path.join(
            EXPERIMENTS_PATH, model_path, 'test_predictions.csv')

        print(
            f"Plotting Precision-Recall curve for {task} using {best_model}...")
        params_file = os.path.join(
            os.path.dirname(test_pred_file), "params.json")

        with open(params_file, "r", encoding="utf-8") as f:
            params = json.load(f)
            is_multilabel = params.get("is_multilabel", True)

        y_true, _, probs = extract_predictions(test_pred_file, is_multilabel)
        precision_recall_curve(y_true, probs, is_multilabel,
                               task, ax, best_model=best_model)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "precision_recall_curves.png"))
    plt.close()
    return best_models_task


def plot_model_metric_all_tasks(task_model_performance, metrics, save_dir):
    """
    Generates a separate multi-panel figure for each metric, where each subplot represents a task
    and displays model performances with proper error bars.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_tasks = len(task_model_performance)

    for metric in metrics:
        ncols = 3
        nrows = math.ceil(num_tasks / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(
            8 * ncols, 5 * nrows), sharex=False)

        axes = axes.flatten()

        for i, (task, model_performance) in enumerate(task_model_performance.items()):
            df_sorted = pd.DataFrame(model_performance).sort_values(
                by=metric, ascending=False).reset_index(drop=True)
            ax = axes[i]
            ax.set_title(f"{task} - {metric}")
            ax.set_xticks(np.arange(len(df_sorted)))
            ax.set_xticklabels(df_sorted["Model"], rotation=45, ha="right")
            ax.set_ylabel(metric)
            ax.set_xlabel("Model")
            ax.set_ylim(0, 1)

            sns.barplot(x="Model", y=metric, hue="Model", data=df_sorted, ax=ax,
                        palette=MODEL_COLORS, legend=False, errorbar=None)

            for index, row in df_sorted.iterrows():
                ax.text(index, row[metric] - 0.2,
                        f"{row[metric]:.3f}", ha="center", color="black")

                # Set error bars
                ci_lower_col = f"{metric} CI Lower"
                ci_upper_col = f"{metric} CI Upper"
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


def find_model_path(task: str, model: str) -> str:
    """Find model checkpoint given the task and model"""
    for exp_dir in os.listdir(EXPERIMENTS_PATH):
        if task not in exp_dir:
            continue
        if model in exp_dir:
            return exp_dir


def collect_metrics_for_task(task: str) -> list[dict]:
    """ Collect or calculate all metrics with confidence interval """
    model_performance = []
    for exp_dir in os.listdir(EXPERIMENTS_PATH):
        if task not in exp_dir:
            continue
        exp_path = os.path.join(EXPERIMENTS_PATH, exp_dir)
        test_pred_file = os.path.join(exp_path, "test_predictions.csv")
        params_file = os.path.join(exp_path, "params.json")

        with open(params_file, "r", encoding="utf-8") as f:
            params = json.load(f)
            is_multilabel = params.get("is_multilabel", True)

        y_true, y_pred, probs = extract_predictions(
            test_pred_file, is_multilabel)
        print(f"Computing metrics for {exp_dir}...")
        metrics = get_metrics_ci(y_true, y_pred)
        model = exp_dir.split("_")[0]
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


def collect_metrics_all_tasks() -> dict:
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
            print(f"Metrics already calculated, loading {outfile}")
            model_performance = pd.read_csv(outfile)
        else:
            model_performance = pd.DataFrame(collect_metrics_for_task(task))
            model_performance.to_csv(outfile, index=False)

        task_model_performance[task] = model_performance
    return task_model_performance


def load_label_mapping(config_path):
    """Loads the id2label mapping from a model's config.json."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return {int(k): v for k, v in config.get("id2label", {}).items()}


def plot_performance_per_label(y_true, y_pred, label_mapping, save_path, task: str, model_name: str):
    """Plots F1, Precision, Recall, and Accuracy per label with multiple rows for each metric and consistent colors."""

    f1s = []
    precisions = []
    recalls = []
    accuracies = []
    sample_counts = []

    if len(y_true.shape) == 1:  # Single-label classification
        for label in label_mapping.keys():
            y_true_binary = (y_true == label).astype(int)
            y_pred_binary = (y_pred == label).astype(int)

            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            f1s.append(f1)

            prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            precisions.append(prec)

            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            recalls.append(recall)

            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            accuracies.append(accuracy)

            sample_count = np.sum(y_true_binary)
            sample_counts.append(sample_count)

    elif len(y_true.shape) == 2:  # Multi-label classification
        for i in range(y_true.shape[1]):
            f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            f1s.append(f1)

            prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            precisions.append(prec)

            recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
            recalls.append(recall)

            accuracy = accuracy_score(y_true[:, i], y_pred[:, i])
            accuracies.append(accuracy)

            sample_count = np.sum(y_true[:, i])
            sample_counts.append(sample_count)

    labels = [label_mapping[label] for label in label_mapping.keys()]

    metrics_df = pd.DataFrame({
        "Label": labels,
        "Sample Count": sample_counts,
        "F1 Score": f1s,
        "Precision": precisions,
        "Recall": recalls,
        "Accuracy": accuracies
    })

    metrics_df.set_index("Label", inplace=True)

    ax = metrics_df.drop('Sample Count', axis=1).plot(kind="bar", figsize=(12, 10), subplots=True, layout=(
        4, 1), legend=True, sharex=True, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j].set_ylabel(metrics_df.columns[i + 1])
            ax[i, j].set_ylim(0, 1)
            for bar in ax[i, j].patches:
                height = bar.get_height()
                ax[i, j].text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}',
                              ha='center', va='bottom')

    for i, label in enumerate(metrics_df.index):
        label_index = ax[0, 0].get_xticks()[i]
        sample_count = metrics_df['Sample Count'].iloc[i]
        ax[0, 0].text(label_index, 1.2, f'#: {int(sample_count)}', ha='center', va='bottom', fontsize=10, color='black')


    plt.suptitle(f"{task} predicted by {model_name} - Performance per Label", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()


def plot_best_f1_scores(directory):
    task_data = []

    # Iterate through CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            task_name = filename.replace("model_performance_", "").replace(".csv", "").replace("_", " ").title()

            # Load CSV
            df = pd.read_csv(os.path.join(directory, filename))

            # Find the row with the highest F1 score
            best_model = df.loc[df['F1'].idxmax()]
            
            # Store relevant data
            task_data.append({
                "task": task_name,
                "model": best_model["Model"],
                "f1_score": best_model["F1"],
                "ci_lower": best_model["F1 CI Lower"],
                "ci_upper": best_model["F1 CI Upper"]
            })

    # Convert to DataFrame
    df_tasks = pd.DataFrame(task_data)

    # Sort by F1 score for better visualization
    df_tasks = df_tasks.sort_values(by="f1_score", ascending=False).reset_index(drop=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar plot without built-in error bars
    sns.barplot(
        x="task",
        y="f1_score",
        hue="model",
        data=df_tasks,
        palette=MODEL_COLORS,
        ax=ax,
        legend=False,
        errorbar=None
    )

    # Add error bars manually
    for index, row in df_tasks.iterrows():
        yerr_lower = row["f1_score"] - row["ci_lower"]
        yerr_upper = row["ci_upper"] - row["f1_score"]
        ax.errorbar(index, row["f1_score"],
                    yerr=[[yerr_lower], [yerr_upper]],
                    fmt='none', color='black', capsize=5)

        # Display values
        ax.text(index, row["f1_score"] - 0.2,
                f"{row['f1_score']:.3f}", ha="center", color="black", fontsize=10)

        ax.text(index, row["ci_lower"] - 0.04,
                f"{row['ci_lower']:.3f}", ha="center", va="bottom", color="black", fontsize=8)

        ax.text(index, row["ci_upper"] + 0.01,
                f"{row['ci_upper']:.3f}", ha="center", va="bottom", color="black", fontsize=8)
        
    # Legend for colors
    for model, color in MODEL_COLORS.items():
        ax.bar(0, 0, color=color, label=model)
    ax.legend(title="Model", loc="upper left", bbox_to_anchor
              =(1, 1), title_fontsize="small")
    
    # Add some padding at bottom so labels are not cut off
    plt.gcf().subplots_adjust(bottom=0.2)

    # Formatting
    ax.set_ylabel("Best F1 Score")
    ax.set_title("Best F1 Score per Task")
    ax.set_xticks(np.arange(len(df_tasks)))
    ax.set_xticklabels(df_tasks["task"], rotation=45, ha="right")
    ax.set_ylim(0, 1)

    plt.show()

def main():
    plot_best_f1_scores('/home/vera/Documents/Arbeit/CRS/PsychNER/model')
    # save_dir = "model/performance_plots"
    # task_model_performance = collect_metrics_all_tasks()
    # plot_model_metric_all_tasks(task_model_performance,
    #                             metrics=["F1", "Accuracy",
    #                                      "Precision", "Recall"],
    #                             save_dir=save_dir)
    # best_models = plot_precision_recall_curve_all_tasks(
    #     task_model_performance, save_dir=save_dir)
    # breakpoint()
    # for task in TASKS:
    #     task_key = task.lower().replace(" ", "_")
    #     best_model = best_models[task_key]
    #     model_path = find_model_path(task_key, best_model)
    #     test_pred_file = os.path.join(
    #         EXPERIMENTS_PATH, model_path, "test_predictions.csv")
    #     checkpoints = [file for file in os.listdir(os.path.join(
    #         EXPERIMENTS_PATH, model_path)) if 'checkpoint' in file]

    #     config_file = os.path.join(
    #         EXPERIMENTS_PATH, model_path, checkpoints[0], "config.json")
    #     label_mapping = load_label_mapping(config_file)
    #     params_file = os.path.join(
    #         os.path.dirname(test_pred_file), "params.json")
    #     with open(params_file, "r", encoding="utf-8") as f:
    #         params = json.load(f)
    #         is_multilabel = params.get("is_multilabel", True)
    #     # Extract predictions
    #     y_true, y_pred, _ = extract_predictions(test_pred_file, is_multilabel)

    #     # Generate per-label performance plots
    #     plot_path = os.path.join(save_dir, f"performance_{task_key}.png")
    #     plot_performance_per_label(
    #         y_true, y_pred, label_mapping, plot_path, task, best_model)

    #     print(f"Saved per-label performance plot for {task}: {plot_path}")


if __name__ == "__main__":
    main()
