import json
import os
from collections import defaultdict
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from iaa import calculate_cohen_kappa_from_cfm_with_ci, calculate_krippendorff_alpha
from itertools import combinations

# Below has to be adjusted given prodigy iteration
FIXED_COLUMNS = ['id', 'text']
# FIXED_COLUMNS = ['id', 'text', 'annotation']
# TODO: better solution for fixed columns


class ProdigyDataReader:
    def __init__(self, jsonl_path: str, annotator: str = None):
        self.jsonl_path = jsonl_path
        self.span_labels = []
        self.tasks = {}
        self.id_to_class_label = {}
        self.annotator = annotator

        self.prodigy_id_to_label = self.get_prodigy_label_map()
        self.df = self._initiate_df()
        self.rejected = []

        self._read_all()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self):
            result = self.df.iloc[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id: str):
        return self.df[self.df['id'] == id]

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

    def export_to_csv(self, out_path: str) -> None:
        # check if path exists
        dir = os.path.dirname(out_path)
        if not os.path.exists(dir):
            raise ValueError(f'Path "{dir}" does not exist')
        else:
            self.df.to_csv(out_path, index=False)

    def get_onehot_task_df(self, task_name: str, one_hot: bool = False) -> pd.DataFrame:
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
            int_to_label = {index: label for index,
                            label in enumerate(self.tasks[task_name])}
            label_to_int = {label: index for index,
                            label in enumerate(self.tasks[task_name])}
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

    def visualize_distribution(self, x_label: str = None, save_path: str = None) -> None:
        if x_label is None:
            classification_tasks = self.get_classification_tasks().keys()

            # Set up the subplots
            num_tasks = len(classification_tasks)
            num_rows = num_tasks // 2 + num_tasks % 2  # 2 columns, variable number of rows
            fig, axes = plt.subplots(
                nrows=num_rows, ncols=2, figsize=(12, 6 * num_rows))
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

    def _is_task_multi_label(self, task_name: str) -> bool:
        if self._is_valid_task(task_name):
            _, df = self.get_label_task_df(task_name)
            for _, row in df.iterrows():
                if len(row[task_name]) > 1:
                    return True
            return False

    def _plot_task_distribution(self, task: str, ax=None):
        df = self.get_onehot_task_df(task)

        relevant_columns = [
            col for col in df.columns if col not in FIXED_COLUMNS]
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
                            ha='center', va='center',
                            xytext=(0, -10),
                            textcoords='offset points')
            else:
                ax.annotate(format(height, '.0f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 5),
                            textcoords='offset points')

    def _is_valid_task(self, task_name: str) -> Union[bool, None]:
        if not task_name in self.get_classification_tasks().keys():
            raise ValueError(
                f'Invalid task name, options are ´{self.get_classification_tasks().keys()}´')
        else:
            return True

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
                if not class_ids and data['answer'] == 'reject':
                    self.rejected.append(data['record_id'])
                else:
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


class ProdigyIAAHelper():
    def __init__(self, list_of_files: list[str], names: list[str] = None) -> None:
        self.prodigy_files = list_of_files
        self.prodigy_readers = []
        for file, name in zip(list_of_files, names):
            prodigy_reader = ProdigyDataReader(file, name)
            self.prodigy_readers.append(prodigy_reader)

        self.tasks = {}
        # sanity checks
        self._inspect_rejected()
        self._check_number_of_samples()
        self._check_order_of_samples()
        self._check_tasks()
        print('Sanity checks passed')

    def _inspect_rejected(self) -> None:
        rejected = []
        for reader in self.prodigy_readers:
            rejected.append(reader.rejected)

        sets_rejected = [set(rej) for rej in rejected]

        agreed_rejected = sets_rejected[0].intersection(*sets_rejected[1:])
        not_agreed_rejected = sets_rejected[0].union(
            *sets_rejected[1:]) - agreed_rejected
        all_rejected = agreed_rejected.union(not_agreed_rejected)

        print(f'Agreed rejected: {agreed_rejected}')
        print(f'Not agreed rejected: {not_agreed_rejected}')

        # remove agreed and not agreed rejected
        for reader in self.prodigy_readers:
            reader.df = reader.df[~reader.df['id'].isin(all_rejected)]

    def _check_number_of_samples(self) -> None:
        # check if all files have the same number of samples
        num_samples_first = len(self.prodigy_readers[0].df)
        for reader in self.prodigy_readers[1:]:
            num_samples = len(reader.df)
            if num_samples != num_samples_first:

                # raise value error the unmatching number and name of data reader
                raise ValueError(
                    f'Number of samples is not the same in {reader.jsonl_path} and {self.prodigy_readers[0].jsonl_path}\n{num_samples} vs {num_samples_first} samples')

    def _check_order_of_samples(self) -> None:
        # TODO: fix hardcoded name of id column
        ids_first = self.prodigy_readers[0].df['id'].to_list()
        # check if all ids are the same
        for reader in self.prodigy_readers[1:]:
            ids = reader.df['id'].to_list()
            if ids != ids_first:
                reader.jsonl_path
                raise ValueError(
                    f'ids are not the same in {reader.jsonl_path} and {self.prodigy_readers[0].jsonl_path}')

    def _check_tasks(self) -> None:
        # check if all tasks are the same
        tasks_first = self.prodigy_readers[0].get_classification_tasks()
        for reader in self.prodigy_readers[1:]:
            tasks = reader.get_classification_tasks()
            if tasks != tasks_first:
                raise ValueError(
                    f'Tasks are not the same in {reader.jsonl_path} and {self.prodigy_readers[0].jsonl_path}')
        self.tasks = tasks_first

    def agreement_per_task(self, task, readers: list[ProdigyDataReader] = None) -> tuple[str, float, float]:
        if self._is_valid_task(task):
            # TODO: refine so that task specific table is not created twice
            if self._task_is_multi_label(task):
                nltk_data = self.reshape_data_for_nltk(task)
                alpha = calculate_krippendorff_alpha(nltk_data)
                return "Krippendorff's Alpha", alpha, 0.0
            else:
                readers = self.prodigy_readers if readers is None else readers
                if len(readers) == 2:
                    int_to_label, reader1_task_df = readers[0].get_label_task_df(
                        task)
                    int_to_label, reader2_task_df = readers[1].get_label_task_df(
                        task)

                    # flatten the pandas series
                    reader1_task_list = reader1_task_df[task].apply(
                        lambda x: x[0] if x else None)
                    reader2_task_list = reader2_task_df[task].apply(
                        lambda x: x[0] if x else None)
                    # replace None with -1
                    reader1_task_list.fillna(-1, inplace=True)
                    reader2_task_list.fillna(-1, inplace=True)
                    # add -1 to int_to_label
                    int_to_label[-1] = 'None'

                    cm = confusion_matrix(
                        reader1_task_list, reader2_task_list, labels=list(int_to_label.keys()))

                    # dist = ConfusionMatrixDisplay(
                    #     cm, display_labels=int_to_label.values())
                    # # turn x axis labels by 45 degrees
                    # dist.plot(xticks_rotation='vertical')

                    cohen, ci = calculate_cohen_kappa_from_cfm_with_ci(cm)

                    return "Cohen's Kappa", cohen, ci
                else:
                    # get all possible combinations of annotators
                    pair_list = list(combinations(self.prodigy_readers, 2))
                    average_kappa = 0
                    average_boundary_limits = 0

                    for c in pair_list:
                        _, kappa, boundary_limits = self.agreement_per_task(
                            task, c)
                        average_kappa += kappa
                        average_boundary_limits += boundary_limits

                    average_kappa /= len(pair_list)
                    average_boundary_limits /= len(pair_list)

                    return "Avg Pairwise Cohen's Kappa", average_kappa, average_boundary_limits

    def agreement_all_tasks(self, pprint: bool = True, csv_path: str = None) -> None:
        agreement_data = []
        for task in self.tasks.keys():
            measure, value, ci_boundary = self.agreement_per_task(task)
            quality = ''
            if measure == "Krippendorff's Alpha":
                if value >= 0.8:
                    quality = 'good'
                elif value >= 0.667:
                    quality = 'acceptable'
                else:
                    quality = 'work required'
                
            elif 'Cohen' in measure:
                if value > 0.8:
                    quality = 'almost perfect'
                elif value > 0.6:
                    quality = 'substantial'
                else:
                    quality = 'work required'
                
            agreement_data.append({'Task': task, 'Agreement Measure': measure,
                                  'Value': value, 'Confidence Interval Boundary': ci_boundary,
                                  'Quality': quality})
            if pprint:
                print(f'{task} - {measure}: {value}, CI boundary: {ci_boundary} --> {quality}')
        # Creating DataFrame
        df = pd.DataFrame(agreement_data)

        # Writing DataFrame to CSV file
        df.to_csv(csv_path, index=False)

    def _is_valid_task(self, task_name: str) -> Union[bool, None]:
        if not task_name in self.tasks.keys():
            raise ValueError(
                f'Invalid task name, options are ´{self.get_classification_tasks().keys()}´')
        else:
            return True

    def _task_is_multi_label(self, task_name: str) -> bool:
        multilabel = False
        for reader in self.prodigy_readers:
            if reader._is_task_multi_label(task_name):
                multilabel = True
                break
        return multilabel

    def reshape_data_for_nltk(self, task_name: str) -> list[tuple[str, str, frozenset]]:
        if self._is_valid_task(task_name):
            nltk_data = []
            for reader in self.prodigy_readers:
                id_to_label, task_dfs = reader.get_label_task_df(task_name)
                # iterate through dataframe
                for _, row in task_dfs.iterrows():
                    new_item = (
                        reader.annotator,
                        str(row['id']),
                        frozenset(row[task_name])
                    )
                    nltk_data.append(new_item)
            return nltk_data

            # iterate through the dataframes in parallel using zip


def calculate_agreement():
    files = [
        '/home/vera/Documents/Arbeit/CRS/PsychNER/data/prodigy_exports/prodigy_export_ben_50_20240418_20240501_181325.jsonl',
        '/home/vera/Documents/Arbeit/CRS/PsychNER/data/prodigy_exports/prodigy_export_pia_50_20240418_20240509_110412.jsonl',
        '/home/vera/Documents/Arbeit/CRS/PsychNER/data/prodigy_exports/prodigy_export_bernard_50_20240418_20240516_091455.jsonl',
        '/home/vera/Documents/Arbeit/CRS/PsychNER/data/prodigy_exports/prodigy_export_julia_50_20240418_20240516_133214.jsonl'
        ]

    names = ['ben', 'pia', 'bernard', 'julia']

    prodigy_aai = ProdigyIAAHelper(files, names)

    prodigy_aai.prodigy_readers[0].export_to_csv(
        'data/annotated_data/prodigy_export_50_20240418/50_ben_flat.csv')
    prodigy_aai.prodigy_readers[1].export_to_csv(
        'data/annotated_data/prodigy_export_50_20240418/50_pia_flat.csv')

    prodigy_aai.agreement_all_tasks(csv_path='data/iaa/task_iaa_stats.csv')


def calculate_pairwise_agreement():
    files = [
        '/home/vera/Documents/Arbeit/CRS/PsychNER/data/prodigy_exports/prodigy_export_ben_50_20240418_20240501_181325.jsonl',
        '/home/vera/Documents/Arbeit/CRS/PsychNER/data/prodigy_exports/prodigy_export_pia_50_20240418_20240509_110412.jsonl',
        '/home/vera/Documents/Arbeit/CRS/PsychNER/data/prodigy_exports/prodigy_export_bernard_50_20240418_20240516_091455.jsonl',
        '/home/vera/Documents/Arbeit/CRS/PsychNER/data/prodigy_exports/prodigy_export_julia_50_20240418_20240516_133214.jsonl'
        ]

    names = ['ben', 'pia', 'bernard', 'julia']
    
    for file, name in zip(files[1:], names[1:]):
        pairwise_file = [files[0], file]
        pairwise_name = [names[0], name]
        reader = ProdigyIAAHelper(pairwise_file, pairwise_name)
        reader.agreement_all_tasks(csv_path=f'data/iaa/task_iaa_stats_{pairwise_name[0]}_{pairwise_name[1]}_23052024.csv')


def main():
    calculate_agreement()
    calculate_pairwise_agreement()


if __name__ == '__main__':
    calculate_agreement()
