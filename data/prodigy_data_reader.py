import json
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

# Below has to be adjusted given prodigy iteration
FIXED_COLUMNS = ['id', 'text']
# FIXED_COLUMNS = ['id', 'text', 'annotation']

class ProdigyDataReader:
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path
        self.span_labels = []
        self.tasks = {}
        self.id_to_class_label = {}

        self.prodigy_id_to_label = self.get_prodigy_label_map()
        self.df = self._initiate_df()
        
        self._read_all()

    def get_prodigy_label_map(self) -> dict:
        if not self.id_to_class_label:
            id_to_class_label = {}
            with open(self.jsonl_path, 'r', encoding='utf-8') as infile:
                line = infile.readline().strip()
                data = json.loads(line)

                # Get the class labels
                options = data['options']
                for options in options:
                    id_to_class_label[options['id']] = options['text']
            self.id_to_class_label = id_to_class_label

        return self.id_to_class_label
    
    def get_classification_tasks(self) -> dict[list[str]]:
        if not self.tasks:
            # subtract the fixed columns
            columns = list(self.df.columns)
            for fixed_col in FIXED_COLUMNS:
                columns.remove(fixed_col)
            tasks = {}
            for col in columns:
                task_group, labels = col.split(': ')
                if task_group not in tasks.keys():
                    tasks[task_group] = []
                tasks[task_group].append(labels)
            self.tasks = tasks

        return self.tasks

    def _encode_class_labels(self) -> None:
        pass

    def _initiate_df(self) -> pd.DataFrame:
        class_labels = ['id', 'text']
        class_labels += list(self.prodigy_id_to_label.values())
        return pd.DataFrame(columns=class_labels)

    def _read_all(self) -> None:
        all_rows = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                data = json.loads(line)
                new_row = self._new_empty_row()
                # get class labels:
                class_ids = data['accept']
                new_row['text'] = data['text']
                new_row['id'] = data['record_id']
                for class_id in class_ids:
                    # set the class label to 1
                    class_label = self.prodigy_id_to_label[class_id]
                    new_row[class_label] = 1
                all_rows.append(new_row)
        self.df = pd.concat([self.df, pd.DataFrame(all_rows)])

    def _new_empty_row(self) -> dict:
        column_names = list(self.df.columns)
        return {col: 0 for col in column_names}

    def export_to_csv(self, out_path: str) -> None:
        self.df.to_csv(out_path, index=False)

    def get_task_df(self, task_name: str, one_hot: bool=False) -> pd.DataFrame:
        # check if class name is valid
        task_names = self.get_classification_tasks().keys()
        
        if task_name not in task_names:
            raise ValueError(f'Invalid class name, options are ´{self.get_classification_tasks().keys()}´')
        else:
            task_filtered = {}
            # add fixed columns
            for fixed_col in FIXED_COLUMNS:
                task_filtered[fixed_col] = self.df[fixed_col]
            # iterate through
            for col in self.df.columns:
                if task_name in col:
                    label = col.split(': ')[1]
                    task_filtered[label] = self.df[col]
            
            task_filtered_df = pd.DataFrame(task_filtered)
        return task_filtered_df
    
    @staticmethod
    def visualize_distribution(df: pd.DataFrame, x_label:str=None, save_path:str=None) -> None:
        # remove fixed columns
        relevant_columns = [col for col in df.columns if col not in FIXED_COLUMNS]
        df = df[relevant_columns]
        
        # Plot the bar plot
        # Sum along the rows to get the total count for each study type
        df_sum = df.sum()

        # Plot the bar plot
        sns.set_theme(style="whitegrid")  # Set style
        ax = sns.barplot(x=df_sum.index, y=df_sum.values)  # Set ci=None to remove error bars

        # Add labels and title
        ax.set_ylabel('Count')
        if x_label:
            ax.set_xlabel(x_label)
            ax.set_title(f'Counts of Different {x_label}')
        # Rotate the x-axis labels
        plt.xticks(rotation=45)
        
        if not save_path:
            plt.show()
        else:
            plt.savefig(save_path)


                
           
if __name__ == '__main__':
    pass
