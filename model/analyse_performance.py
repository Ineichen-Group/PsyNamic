import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
from confidenceinterval.bootstrap import bootstrap_ci

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def make_model_comparison_plot():
    models = [
        "pubmedbert",
        "biomedbert-abstract",
        "scibert",
        "biobert",
        "clinicalbert",
        "biolinkbert"
    ]

    # List of tasks
    tasks = [
        "Data Collection", "Data Type", "Number of Participants", "Age of Participants", "Application Form",
        "Clinical Trial Phase", "Condition", "Outcomes", "Regimen", "Setting", "Study Control", "Study Purpose",
        "Substance Naivety", "Substances", "Sex of Participants", "Study Conclusion", "Study Type", "NER Bio",
        "Relevant"
    ]

    experiments_path = '/home/vebern/scratch/PsyNamic/model/experiments'
    date = '20250221'

    task_model_performance = {}
    for task in tasks:
        task_ident = task.replace(' ', '_').lower()
        model_performance = {}
        for model in models:
            exp_dir = f'{model}_{task_ident}_{date}'
            exp_dir_path = os.path.join(experiments_path, exp_dir)
            eval_file = os.path.join(exp_dir_path, 'test_eval.csv')
            if task == 'Substances':
                continue
            if task == 'NER Bio':
                continue
            if not os.path.exists(eval_file):
                print(f'{eval_file} does not exist')
                continue
            with open(eval_file, 'r', encoding='utf-8') as file:
                try:
                    eval_data = json.load(file)
                except json.JSONDecodeError:
                    file.seek(0)
                    content = file.read()
                    json_objects = re.findall(r'{[^{}]*}', content)
                    data_objects = [json.loads(obj) for obj in json_objects]
                    eval_data = data_objects[0]
                model_performance[model] = eval_data['test_f1']
        task_model_performance[task] = model_performance

    # Create a figure and axes for the subplots
    num_tasks = len(task_model_performance)
    num_columns = 3  # You can adjust this to change the number of columns in the grid
    num_rows = (num_tasks // num_columns) + \
        (1 if num_tasks % num_columns != 0 else 0)

    # Set the size of the figure
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 6 * num_rows))

    # Flatten axes array for easier iteration
    axes = axes.flatten()

    # Define custom colors for each model (optional)
    model_colors = {
        "pubmedbert": "#1f77b4",  # Blue
        "biomedbert-abstract": "#ff7f0e",  # Orange
        "scibert": "#2ca02c",  # Green
        "biobert": "#d62728",  # Red
        "clinicalbert": "#9467bd",  # Purple
        "biolinkbert": "#8c564b"  # Brown
    }

    # Loop through each task and create a bar plot for each task
    for i, (task, model_performance) in enumerate(task_model_performance.items()):
        task_data = []
        
        for model, f1_score_value in model_performance.items():
            exp_dir = f'{model}_{task.replace(" ", "_").lower()}_{date}'
            exp_dir_path = os.path.join(experiments_path, exp_dir)
            test_pred_file = os.path.join(exp_dir_path, 'test_predictions.csv')
            
            if not os.path.exists(test_pred_file):
                print(f"Missing predictions file: {test_pred_file}")
                continue
            
            # Get F1 score and confidence interval
            score, ci = get_f1_ci(test_pred_file)
            ci_lower, ci_upper = ci

            # check if two f1 scores are equal
            if score != f1_score_value:
                raise ValueError(f"Computed F1 score {score} does not match the value in the evaluation file {f1_score_value}")

            task_data.append({
                'Model': model,
                'F1 Score': score,
                'CI Lower': ci_lower,
                'CI Upper': ci_upper
            })
        
        # Convert to DataFrame
        task_df = pd.DataFrame(task_data)

        if task_df.empty:
            print(f"Skipping {task} because it has no valid data.")
            continue

        max_model = task_df.loc[task_df['F1 Score'].idxmax(), 'Model']

        # Create a color mapping where the highest model gets its original color, and others are grey
        bar_colors = [model_colors[model] if model == max_model else "grey" for model in task_df['Model']]

        # Calculate error bars (upper - lower)
        ci_errors = task_df['CI Upper'] - task_df['F1 Score']

        # Plot with custom colors and error bars
        sns.barplot(x='Model', y='F1 Score', hue='Model', data=task_df, ax=axes[i],
                    palette=bar_colors, legend=False, errorbar=None)

        # Add error bars manually
        axes[i].errorbar(x=range(len(task_df)), y=task_df['F1 Score'],
                        yerr=ci_errors, fmt='none', capsize=5, color='black', alpha=0.7)

        # Add value labels on top of bars
        for container in axes[i].containers:
            for p in container:
                height = p.get_height()
                if height > 0:
                    axes[i].text(p.get_x() + p.get_width() / 2., height + 0.02,
                                f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='black')

        axes[i].set_title(task)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylabel('F1 Score')
        axes[i].set_xlabel('Model')
        axes[i].set_ylim(0, 1)  # Ensure consistent y-axis across plots


    # Hide any unused axes (if num_tasks is less than the number of subplots)
    for j in range(num_tasks, len(axes)):
        axes[j].axis('off')

    # Adjust layout to make sure the labels don't overlap
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('model_performance_task_plots.png',
                bbox_inches='tight')  # Save to PNG file


def get_f1_ci(test_pred_file: str) -> tuple[float, tuple[float, float]]:
    pred_df = pd.read_csv(test_pred_file, encoding='utf-8')

    y_true = pred_df['label']
    y_pred = pred_df['prediction']
    score, ci = bootstrap_ci(y_true=y_true,
                        y_pred=y_pred,
                        metric=f1_score,
                        confidence_level=0.95,
                        n_resamples=9999,
                        method='bootstrap_bca',
                        random_state=42)


    return score, ci
