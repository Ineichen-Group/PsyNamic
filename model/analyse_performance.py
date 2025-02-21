import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

experiments_path = 'model/experiments'
date = '20250221'

task_model_performance = {}
for task in tasks:
    task_ident = task.replace(' ', '_').lower()
    model_performance = {}
    for model in models:
        exp_dir = f'{model}_{task_ident}_{date}'
        exp_dir_path = os.path.join(experiments_path, exp_dir)
        if not os.path.exists(exp_dir_path):
            raise ValueError(f'Experiment directory {exp_dir_path} does not exist')
        
        eval_file = os.path.join(exp_dir_path, 'test_eval.csv')
        with open(eval_file, 'r') as file:
            eval_data = json.load(file)
            model_performance[model] = eval_data['f1']

    task_model_performance[task] = model_performance

# Convert the data into a pandas DataFrame for easy plotting
data = []

for task, model_performance in task_model_performance.items():
    for model, f1_score in model_performance.items():
        data.append({
            'Task': task,
            'Model': model,
            'F1 Score': f1_score
        })

df = pd.DataFrame(data)

# Define a consistent color palette for the models
palette = sns.color_palette("Set2", len(models))  # You can choose any palette you prefer

# Plot separate bar graphs for each task
n_tasks = len(tasks)
n_cols = 3  # Number of columns per row of plots
n_rows = (n_tasks // n_cols) + (n_tasks % n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()  # Flatten the axes array to make it easier to work with

# Create a barplot for each task
for i, task in enumerate(tasks):
    task_data = df[df['Task'] == task]
    sns.barplot(x='Model', y='F1 Score', data=task_data, ax=axes[i], palette=palette)
    axes[i].set_title(f'Performance for {task}')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

# Hide any unused axes
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
