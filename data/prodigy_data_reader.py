import json
import os
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from typing import Union
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, multilabel_confusion_matrix
from stride_utils.iaa import (calculate_cohen_kappa_from_cfm_with_ci,
                              calculate_krippendorff_alpha_with_ci,
                              interpret_alpha, interpret_kappa)
from sklearn.preprocessing import MultiLabelBinarizer
# Below has to be adjusted given prodigy iteration
FIXED_COLUMNS = ['id', 'text']
# FIXED_COLUMNS = ['id', 'text', 'annotation']
# TODO: better solution for fixed columns

# Check if task option are properly sorted


class ProdigyDataReader:
    def __init__(self, jsonl_path: str, annotator: str = None, thematic_split: bool = True):
        self.jsonl_path = jsonl_path
        self._check_path()

        self.span_labels = []
        self._tasks = {}

        # Used with abstract only appearing once
        self._id_to_class_label = {}
        # User with abstract appearing three times
        self._thematic_id_to_class_label = {}

        self.annotator = annotator
        self.thematic_split = thematic_split

        # Check if data has been reordered
        if self.thematic_split and 'reordered' not in self.jsonl_path:
            self._check_order()

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

    def get_prodigy_label_map(self, thematic: str = None) -> dict:
        # Case 1: Thematic split: abstracts appear three times & label mapping has been collected already
        if self.thematic_split:
            # Case
            if self._thematic_id_to_class_label:
                return self._thematic_id_to_class_label[thematic] if thematic else self._thematic_id_to_class_label
        # Case 2: Abstracts appear only once & label mapping has been collected already
        else:
            if self._id_to_class_label:
                return self._id_to_class_label

        # Case 3: Collect label mapping

        with open(self.jsonl_path, 'r', encoding='utf-8') as infile:
            lines = [infile.readline().strip() for _ in range(
                3)] if self.thematic_split else [infile.readline().strip()]

            for line in lines:
                id_to_class_label = {}
                data = json.loads(line)
                # Get the class labels
                options = data['options']
                for options in options:
                    id_to_class_label[options['id']] = options['text']
                if self.thematic_split:
                    thematic_name = data['annotation']
                    self._thematic_id_to_class_label[thematic_name] = id_to_class_label
                else:
                    self._id_to_class_label = id_to_class_label

        return self.get_prodigy_label_map(thematic)

    def get_classification_tasks(self) -> dict[list[str]]:
        if not self._tasks:
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
            # Order the keys
            tasks = dict(sorted(tasks.items()))
            # Order the values
            for key in tasks.keys():
                tasks[key] = sorted(tasks[key])
            self._tasks = tasks

        return self._tasks

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
                            label in enumerate(self._tasks[task_name])}
            label_to_int = {label: index for index,
                            label in enumerate(self._tasks[task_name])}
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
        label_mapping = self.get_prodigy_label_map()
        if self.thematic_split:
            for thematic, thematic_label_mapping in label_mapping.items():
                class_labels += list(thematic_label_mapping.values())
        else:
            class_labels += list(label_mapping.values())
        return pd.DataFrame(columns=class_labels)

    def _read_all(self) -> None:
        all_rows = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as infile:
            lines = []
            for line_dict in infile:
                line_dict = json.loads(line_dict)
                lines.append(line_dict)
                # keep collecting all information until the next abstract
                if self.thematic_split:
                    if len(lines) < 3:
                        continue
                    else:
                        # check if the three lines are from the same abstract
                        ids = [line['record_id'] for line in lines]
                        if len(set(ids)) != 1:
                            raise ValueError(
                                f'Thematically split information about abstract is not in order; i.e. different ids have been found:  {ids}')

                new_row = self._new_empty_row()
                rejected = []
                for line_dict in lines:
                    if self.thematic_split:
                        thematic = line_dict['annotation']
                        prodigy_label_map = self.get_prodigy_label_map(
                            thematic)
                    else:
                        prodigy_label_map = self.get_prodigy_label_map()
                    # get class labels:
                    class_ids = line_dict['accept']
                    new_row['text'] = line_dict['text']
                    new_row['id'] = line_dict['record_id']

                    if not class_ids and line_dict['answer'] == 'reject':
                        rejected.append(line_dict['record_id'])
                    else:
                        for class_id in class_ids:
                            # set the class label to 1
                            class_label = prodigy_label_map[class_id]
                            new_row[class_label] = 1

                # check if the same abstract was accepted and rejected -> annotation error
                if self.thematic_split and (len(rejected) == 1 or len(rejected) == 2):
                    raise ValueError(
                        f'Same abstract was accepted and rejected {rejected[0]}')

                all_rows.append(new_row)
                if rejected:
                    self.rejected.append(rejected[0])

                # reset for next abstract
                lines = []
                rejected = []

        self.df = pd.concat([self.df, pd.DataFrame(all_rows)])
        # reorder rows according to the id
        self.df = self.df.sort_values(by='id')

    def _new_empty_row(self) -> dict:
        column_names = list(self.df.columns)
        return {col: 0 for col in column_names}

    def _check_order(self) -> None:
        ordered_file = self.jsonl_path.replace('.jsonl', '_reordered.jsonl')
        records = {}
        with open(self.jsonl_path, 'r') as infile:
            for line in infile:
                entry = json.loads(line)
                record_id = entry['record_id']

                if record_id not in records:
                    records[record_id] = []
                records[record_id].append(entry)

        # Check that each abstract has exactly three entries
        for record_id, entries in records.items():
            if len(entries) != 3:
                raise ValueError(
                    f"Record ID {record_id} does not have exactly three entries")

        sorted_record_ids = sorted(records.keys())

        # Write the sorted entries to the output JSONL file
        with open(ordered_file, 'w') as outfile:
            for record_id in sorted_record_ids:
                for entry in records[record_id]:
                    outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
        self.jsonl_path = ordered_file

    def _check_path(self) -> None:
        if not os.path.exists(self.jsonl_path):
            raise ValueError(f'Path "{self.jsonl_path}" does not exist')
        # Check if there is a reordered version
        filename, file_extension = os.path.splitext(self.jsonl_path)
        filename_reorder = filename + '_reordered' + file_extension
        if os.path.exists(filename_reorder):
            self.jsonl_path = filename_reorder


class ProdigyIAAHelper():
    def __init__(self, list_of_files: list[str], names: list[str] = None, log: str = 'iaa.log', thematic_split: bool = True) -> None:
        self.prodigy_files = list_of_files
        self.prodigy_readers = []
        self.log = log
        self.names = names if names else [
            f'Annotator_{i}' for i in range(len(list_of_files))]

        for file, name in zip(list_of_files, names):
            prodigy_reader = ProdigyDataReader(file, name, thematic_split)
            self.prodigy_readers.append(prodigy_reader)

        self._inital_log()
        self.tasks = {}
        # sanity checks
        self._inspect_rejected()
        self._check_number_of_samples()
        self._check_order_of_samples()
        self._check_tasks()
        print('Sanity checks passed')

    def _inital_log(self):
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log, 'w') as f:
            f.write('Log file for IAA calculations\n')
            f.write(f'Created at {date}\n')
            f.write('Datafiles used:\n')
            for file, name in zip(self.prodigy_files, self.names):
                f.write(f'{file} ({name})\n')

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

        # write to log
        with open(self.log, 'a') as f:
            f.write(f'Agreed rejected: {agreed_rejected}\n')
            f.write(f'Not agreed rejected: {not_agreed_rejected}\n')

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

    def get_task_df(self, task: str) -> pd.DataFrame:
        if self._is_valid_task(task):
            reader_task_df_list = []
            int_to_label_list = []
            for reader in self.prodigy_readers:
                int_to_label, reader_task_df = reader.get_label_task_df(task)
                reader_task_df['reader'] = reader.annotator
                reader_task_df_list.append(reader_task_df)
                int_to_label_list.append(int_to_label)

            # check if the int_to_label is the same for all readers
            if not all(int_to_label == int_to_label_list[0] for int_to_label in int_to_label_list):
                raise ValueError(
                    f'Label mapping is not the same in {self.prodigy_readers[0].jsonl_path} and {self.prodigy_readers[1].jsonl_path}')
            concat_task_df = pd.concat(
                reader_task_df_list, ignore_index=True)
            return int_to_label_list[0], concat_task_df

    def agreement_per_task(self, task: str, readers: list[ProdigyDataReader] = None) -> tuple[str, float, float]:
        if self._is_valid_task(task):
            # TODO: refine so that task specific table is not created twice
            if self._task_is_multi_label(task):
                int_to_label, task_df = self.get_task_df(task)
                alpha, low, high = calculate_krippendorff_alpha_with_ci(
                    task_df, 'id', 'reader', task)
                confidence_interval = high - low
                return "Krippendorff's Alpha", alpha, confidence_interval
            else:
                readers = self.prodigy_readers if readers is None else readers
                if len(readers) == 2:
                    int_to_label, reader1_task_df = readers[0].get_label_task_df(
                        task)
                    int_to_label, reader2_task_df = readers[1].get_label_task_df(
                        task)
                    # TODO: unduplicate code
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
            self.task_confusion_matrix(task)
            quality = interpret_alpha(
                value) if measure == "Krippendorff's Alpha" else interpret_kappa(value)

            agreement_data.append({'Task': task, 'Agreement Measure': measure,
                                  'Value': value, 'Confidence Interval Boundary': ci_boundary,
                                   'Quality': quality})
            if pprint:
                print(
                    f'{task} - {measure}: {value}, CI boundary: {ci_boundary} --> {quality}')
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

    def task_confusion_matrix(self, task_name: str):
        if len(self.prodigy_readers) != 2:
            raise NotImplementedError(
                'Only pairwise comparison is implemented for confusion matrix')
        else:
            if self._is_valid_task(task_name):
                id_to_label, task_df = self.get_task_df(task_name)
                # where annotator = name[0], order by id, make array
                first_pred = task_df[task_df['reader'] == self.names[0]].sort_values(by='id')[
                    task_name]
                second_pred = task_df[task_df['reader'] == self.names[1]].sort_values(by='id')[
                    task_name]

                if self._task_is_multi_label(task_name):
                    first_pred_matrix = MultiLabelBinarizer(classes=list(
                        id_to_label.keys())).fit_transform(first_pred.to_numpy())
                    second_pred_matrix = MultiLabelBinarizer(classes=list(
                        id_to_label.keys())).fit_transform(second_pred.to_numpy())
                    cm = multilabel_confusion_matrix(
                        first_pred_matrix, second_pred_matrix, labels=list(id_to_label.keys()))
                    self.plot_multilabel_confusion_matrix(
                        cm, list(id_to_label.values()))

                else:

                    first_pred_flat = first_pred.apply(
                        lambda x: x[0] if x else None)
                    second_pred_flat = second_pred.apply(
                        lambda x: x[0] if x else None)
                    # if there are None values, replace them with -1
                    first_pred_flat.fillna(-1, inplace=True)
                    second_pred_flat.fillna(-1, inplace=True)
                    # add -1 to the label mapping
                    id_to_label[-1] = 'None'

                    cm = confusion_matrix(
                        first_pred_flat, second_pred_flat, labels=list(id_to_label.keys()))
                    # show confusion matrix
                    dist = ConfusionMatrixDisplay(
                        cm, display_labels=id_to_label.values())
                    dist.plot(xticks_rotation='vertical')
                    # add title
                    plt.title(f'{task_name} - Single Label')
                    plt.show()

    @staticmethod
    def plot_multilabel_confusion_matrix(cm, labels, task_name: str = None):
        """
        Plots the multilabel confusion matrix.

        Parameters:
        cm (ndarray): The multilabel confusion matrix.
        labels (list): The list of labels.
        """
        n_labels = len(labels)
        fig, axes = plt.subplots(n_labels, 1, figsize=(10, 5 * n_labels))

        if n_labels == 1:
            # Ensure axes is a list of axes even if there's only one label
            axes = [axes]

        for i, (ax, label) in enumerate(zip(axes, labels)):
            # Extract the true negatives, false positives, false negatives, and true positives
            tn, fp, fn, tp = cm[i].ravel()

            # Create the confusion matrix for the current label
            matrix = np.array([[tn, fp], [fn, tp]])

            # Plot the confusion matrix
            cax = ax.matshow(matrix, cmap=plt.cm.Blues)
            fig.colorbar(cax, ax=ax)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Negative', 'Positive'])
            ax.set_yticklabels(['Negative', 'Positive'])
            ax.set_title(f'Confusion Matrix for {label}')

            # Annotate the confusion matrix
            for (j, k), val in np.ndenumerate(matrix):
                ax.text(k, j, f'{val}', ha='center', va='center', color='red')
        # add title
        if task_name:
            plt.suptitle(f'{task_name} - Multi Label')
        plt.tight_layout()
        plt.show()
