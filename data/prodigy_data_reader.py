import json
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union
import os

# Below has to be adjusted given prodigy iteration
FIXED_COLUMNS = ['id', 'text']
# FIXED_COLUMNS = ['id', 'text', 'annotation']
# TODO: better solution for fixed columns

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
        # reorder rows according to the id
        self.df = self.df.sort_values(by='id')

    def _new_empty_row(self) -> dict:
        column_names = list(self.df.columns)
        return {col: 0 for col in column_names}

    def export_to_csv(self, out_path: str) -> None:
        # check if path exists
        dir = os.path.dirname(out_path)
        if not os.path.exists(dir):
            raise ValueError(f'Path "{dir}" does not exist')
        else:
            self.df.to_csv(out_path, index=False)

    def get_onehot_task_df(self, task_name: str, one_hot: bool=False) -> pd.DataFrame:    
        if self._is_valid_task(task_name):
            task_filtered = {}
            # add fixed columns
            for fixed_col in FIXED_COLUMNS:
                task_filtered[fixed_col] = self.df[fixed_col]
            for col in self.df.columns:
                if task_name in col:
                    label = col.split(': ')[1]
                    task_filtered[label] = self.df[col]
            
            task_filtered_df = pd.DataFrame(task_filtered)
        return task_filtered_df
    
    def get_label_task_df(self, task_name: str) -> tuple[dict, pd.DataFrame]:
        if self._is_valid_task(task_name):
            int_to_label = {index: label for index, label in enumerate(self.tasks[task_name])}
            label_to_int = {label: index for index, label in enumerate(self.tasks[task_name])}
            print(label_to_int)
            task_filtered = {}
            # add fixed columns
            for fixed_col in FIXED_COLUMNS:
                task_filtered[fixed_col] = self.df[fixed_col]
            
            # add task column
            task_filtered[task_name] = []
            
            # iterate through rows
            for _, row in self.df.iterrows():
                labels = []
                for col in self.df.columns:
                    if task_name in col:
                        label = col.split(': ')[1]
                        if row[col] == 1:
                            labels.append(label_to_int[label])
                task_filtered[task_name].append(labels)
            
            new_datafame = pd.DataFrame(task_filtered)
            
            return int_to_label, new_datafame
   
    def visualize_distribution(self, x_label:str=None, save_path:str=None) -> None:
        if x_label is None:
            classification_tasks = self.get_classification_tasks().keys()
            
            # Set up the subplots
            num_tasks = len(classification_tasks)
            num_rows = num_tasks // 2 + num_tasks % 2  # 2 columns, variable number of rows            
            fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, 6 * num_rows))
            axes = axes.flatten()  # Flatten axes if more than 1 row
            
            for idx, task in enumerate(classification_tasks):
                ax = axes[idx] if num_rows > 1 else axes
                self._plot_task_distribution(task, ax)
            
            plt.subplots_adjust(hspace=0.8, top=0.96)
            
            fig.suptitle('Overview of all Classification Tasks', fontsize=16)
            
        else:
            self._plot_task_distribution(x_label)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def _plot_task_distribution(self, task:str, ax=None):
        df = self.get_onehot_task_df(task)
        
        relevant_columns = [col for col in df.columns if col not in FIXED_COLUMNS]
        df = df[relevant_columns]

        df_sum = df.sum()
        
        if ax is None:
            ax = plt.gca()
        
        sns.set_theme(style="whitegrid")  # Set style
        sns.barplot(x=df_sum.index, y=df_sum.values, ax=ax)
        
        ax.set_ylabel('Count')
        ax.set_xlabel('')
        ax.set_title(f'Counts of {task}')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(format(height, '.0f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, -10), 
                            textcoords = 'offset points')
            else:
                ax.annotate(format(height, '.0f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 5), 
                            textcoords = 'offset points')

    def _is_valid_task(self, task_name: str) -> Union[bool, None]:
        if not task_name in self.get_classification_tasks().keys():
            raise ValueError(f'Invalid task name, options are ´{self.get_classification_tasks().keys()}´')
        else:
            return True

class ProdigyIAAHelper():
    def __init__(self, list_of_files: list[str]) -> None:
        self.prodigy_files = list_of_files
        self.prodigy_readers = []
        for file in list_of_files:
            self.prodigy_readers.append(ProdigyDataReader(file))
            
        # sanity checks
        self._check_number_of_samples()
        self._check_order_of_samples()
        self._check_tasks()
        print('All checks passed')
    
    def _check_number_of_samples(self) -> None:
        # check if all files have the same number of samples
        num_samples_first = len(self.prodigy_readers[0].df)
        for reader in self.prodigy_readers[1:]:
            num_samples = len(reader.df)
            if num_samples != num_samples_first:
                # raise value error the unmatching number and name of data reader
                raise ValueError(f'Number of samples is not the same in {reader.jsonl_path} and {self.prodigy_readers[0].jsonl_path}\n{num_samples} vs {num_samples_first} samples' )
            
    def _check_order_of_samples(self) -> None:
        # TODO: fix hardcoded name of id column
        ids_first = self.prodigy_readers[0].df['id'].to_list()
        # check if all ids are the same
        for reader in self.prodigy_readers[1:]:
            ids = reader.df['id'].to_list()
            if ids != ids_first:
                reader.jsonl_path
                raise ValueError(f'ids are not the same in {reader.jsonl_path} and {self.prodigy_readers[0].jsonl_path}')
    
    def _check_tasks(self) -> None:
        # check if all tasks are the same
        tasks_first = self.prodigy_readers[0].get_classification_tasks()
        for reader in self.prodigy_readers[1:]:
            tasks = reader.get_classification_tasks()
            if tasks != tasks_first:
                raise ValueError(f'Tasks are not the same in {reader.jsonl_path} and {self.prodigy_readers[0].jsonl_path}')        
    
    def cohen_kappa_per_task(self, task):
        for reader in self.prodigy_readers:
            task = reader.get_label_task_df(task)
        
    
if __name__ == '__main__':
    pass
