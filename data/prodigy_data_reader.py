import json
import os
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from typing import Union, Literal
import numpy as np
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, multilabel_confusion_matrix
from stride_utils.iaa import (calculate_cohen_kappa_from_cfm_with_ci,
                              calculate_krippendorff_alpha_with_ci,
                              calculate_percentage_agreement,
                              interpret_alpha, interpret_kappa)
from sklearn.preprocessing import MultiLabelBinarizer
# Below has to be adjusted given prodigy iteration
FIXED_COLUMNS = ['id', 'text', 'annotator', 'source_file']
# TODO: better solution for fixed columns


class ProdigyDataReader:
    def __init__(self, jsonl_path: str, annotator: str = 'unkown', purpose = Literal['ner', 'class', 'both']) -> None:
        self.jsonl_path = jsonl_path
        self._check_path()
        self.index = 0

        self.span_labels = set()
        self.nr_rejected = 0
        self.nr_total = 0
        self._tasks = {}
        self._thematic_split = None
        self._line_dicts = []

        # Used with abstract only appearing once
        self._id_to_class_label = {}
        # User with abstract appearing three times
        self._thematic_id_to_class_label = {}

        self.annotator = annotator

        # Check if data has been reordered
        if self.has_thematic_split() and 'reordered' not in self.jsonl_path:
            self._check_order()

        self.df = self._initiate_df()
        self.rejected = []

        self.ner_tags = set()
        self.ner_per_abstract = defaultdict(lambda: defaultdict(list))
        
        self.purpose = purpose
        if purpose == 'both':
            self._read_all_class()
            self._read_all_ner()
        elif purpose == 'class':
            self._read_all_class()
        elif purpose == 'ner':
            self._read_all_ner()

        self.df = self.replace_newlines(self.df)
        # TODO: better solution for fixed columns
        self.df[FIXED_COLUMNS[2]] = self.annotator
        self.df[FIXED_COLUMNS[3]] = self.jsonl_path

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self):
            result = self.df.iloc[self.index]
            self.index += 1
            return result
        else:
            self.index = 0
            raise StopIteration

    def __len__(self):
        return len(self.df)

    def __getitem__(self, record_id: str):
        return self.df[self.df['id'] == record_id]

    def __contains__(self, item: Union[int, list[int]]) -> bool:
        if isinstance(item, list):
            return all([i in self.df['id'].to_list() for i in item])
        else:
            return item in self.df['id'].to_list()

    def has_thematic_split(self) -> bool:
        ''' Check if the samples appear 3x times to annotate different tasks (grouped thematicall)'''

        if self._thematic_split is None:
            # Read in first three lines
            with open(self.jsonl_path, 'r', encoding='utf-8') as infile:
                first_three_lines = [json.loads(
                    infile.readline().strip()) for _ in range(3)]
                # check if options are the same in the files
                options1 = first_three_lines[0]['options']
                options2 = first_three_lines[1]['options']
                options3 = first_three_lines[2]['options']
                id1 = first_three_lines[0]['record_id']
                id2 = first_three_lines[1]['record_id']
                id3 = first_three_lines[2]['record_id']

                # Case 1: Three different abstracts with the same options
                if options1 == options2 == options3 and id1 != id2 != id3:
                    self._thematic_split = False
                # Case 2: The same abstract with different options
                elif id1 == id2 == id3 and options1 != options2 != options3:
                    self._thematic_split = True
                # Case 3: Unordered thematic split or invalid format
                else:
                    self._check_order()
                    self.has_thematic_split()
        return self._thematic_split

    def get_prodigy_label_map(self, thematic: str = None) -> dict:
        # Case 1: Thematic split: abstracts appear three times & label mapping has been collected already
        if self.has_thematic_split():
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
                3)] if self.has_thematic_split() else [infile.readline().strip()]

            for line in lines:
                id_to_class_label = {}
                data = json.loads(line)
                # Get the class labels
                options = data['options']
                for options in options:
                    id_to_class_label[options['id']] = options['text']
                if self.has_thematic_split():
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
                try:
                    columns.remove(fixed_col)
                except ValueError:
                    pass
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

    def get_onehot_task_df(self, task_name: str) -> pd.DataFrame:
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

    def get_label_task_df(self, task_name: str, label_to_int: Union[dict[str], None] = None) -> tuple[dict, pd.DataFrame]:
        if self._is_valid_task(task_name):
            if not label_to_int:
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
                    if col.startswith(task_name):
                        label = col.split(': ')[1]
                        if row[col] == 1:
                            labels.append(label_to_int[label])
                task_filtered[task_name].append(labels)

            new_dataframe = pd.DataFrame(task_filtered)
            return label_to_int, new_dataframe

    def are_ids_in_df(self, record_ids: list[int]) -> bool:
        '''Check if all record ids are in the dataframe'''
        return all([i in self.df['id'].to_list() for i in record_ids])

    def remove_ids(self, ids: list[int]) -> None:
        '''Remove rows with the given ids'''
        self.df = self.df[~self.df['id'].isin(ids)]

    def write_jsonl(self, path: str = None) -> None:
        '''Write the dataframe to a jsonl file'''
        # if path not given, overwrite the original file
        if not path:
            path = self.jsonl_path
        with open(path, 'w', encoding='utf-8') as outfile:
            for line_dict in self._line_dicts:
                outfile.write(json.dumps(line_dict, ensure_ascii=False) + '\n')

    @property
    def ids(self) -> list[int]:
        return self.df['id'].to_list()

    def get_ner_per_abstract(self, record_id: int, label: str = None) -> list[(str, str)]:
        if record_id not in self.ids:
            raise ValueError(f'Id {record_id} not found in dataframe')
        ners = []
        if label:
            for ner in self.ner_per_abstract[record_id][label]:
                ners.append((' '.join(ner), label))
        else:
            for label in self.ner_per_abstract[record_id]:
                for ner in self.ner_per_abstract[record_id][label]:
                    ners.append((' '.join(ner), label))
        return ners

    def get_text(self, record_id: int) -> str:
        '''Get the abstract of a given record id'''
        entry = self.df[self.df['id'] == record_id]
        if entry.empty:
            raise ValueError(f'Id {record_id} not found in dataframe')
        else:
            return self.df[self.df['id'] == record_id]['text'].values[0]

    def get_label(self, record_id: int, task: str) -> list[str]:
        '''Get the label of a given record id and task'''
        self._is_valid_task(task)
        entry = self.df[self.df['id'] == record_id]
        if entry.empty:
            raise ValueError(f'Id {record_id} not found in dataframe')
        else:
            labels = []
            # find rows satrtswith task
            for col in self.df.columns:
                if col.startswith(task):
                    label = col.split(': ')[1]
                    if entry[col].values[0] == 1:
                        labels.append(label)
                        # if task is multilabel, return list of labels
                        if not self._is_task_multi_label(task):
                            return labels
            return labels

    def get_labels(self, record_id: int) -> dict[str, list[str]]:
        '''Get all labels of a given record id'''
        entry = self.df[self.df['id'] == record_id]
        if entry.empty:
            raise ValueError(f'Id {record_id} not found in dataframe')
        else:
            labels = {}
            for task in self.get_classification_tasks().keys():
                labels[task] = self.get_label(record_id, task)
            return labels

    def get_ner(self, record_id: int):
        '''Get NER data of a given record id'''
        entry = self.df[self.df['id'] == record_id]
        if entry.empty:
            raise ValueError(f'Id {record_id} not found in dataframe')
        else:
            # check if dict is empty
            if not self.ner_per_abstract[record_id]:
                raise ValueError(f'No NER data found for id {record_id}')
            else:
                return self.ner_per_abstract[record_id]

    def get_ids_with_ner(self, ner_label: str) -> list[int]:
        '''Get all record ids where a specific NER label is present'''
        if not self.ner_tags:
            raise ValueError('No NER data found in dataframe')
        record_ids = []
        for record_id in self.ner_per_abstract.keys():
            for label in self.ner_per_abstract[record_id]:
                if label == ner_label:
                    record_ids.append(record_id)
        return record_ids

    def _is_task_multi_label(self, task_name: str) -> bool:
        '''Check if a task is multi-label'''
        if self._is_valid_task(task_name):
            try:
                return self._task_multilabel[task_name]
            except KeyError:
                _, df = self.get_label_task_df(task_name)
                for _, row in df.iterrows():
                    if len(row[task_name]) > 1:
                        self._task_multilabel[task_name] = True
                        return True
                self._task_multilabel[task_name] = False
                return False

    def _is_valid_task(self, task_name: str) -> Union[bool, None]:
        '''Check if a task actually exists in data'''
        if not task_name in self.get_classification_tasks():
            raise ValueError(
                f"Invalid task name ´{task_name}´, options are ´{self.get_classification_tasks().keys()}´")
        else:
            return True

    def _initiate_df(self) -> pd.DataFrame:
        class_labels = ['id', 'text']
        label_mapping = self.get_prodigy_label_map()
        if self.has_thematic_split():
            for _, thematic_label_mapping in label_mapping.items():
                class_labels += list(thematic_label_mapping.values())
        else:
            class_labels += list(label_mapping.values())
        return pd.DataFrame(columns=class_labels)

    def _read_all_class(self) -> None:
        all_rows = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as infile:
            lines = []
            for line in infile:

                line_dict = json.loads(line)
                self._line_dicts.append(line_dict)
                lines.append(line_dict)
                # keep collecting all information until the next abstract
                if self.has_thematic_split():
                    if len(lines) < 3:
                        continue
                    else:
                        self.nr_total += 1
                        # check if the three lines are from the same abstract
                        ids = [line['record_id'] for line in lines]
                        if len(set(ids)) != 1:
                            raise ValueError(
                                f'Thematically split information about abstract is not in order; i.e. different ids have been found:  {ids}')
                else:
                    self.nr_total += 1

                new_row = self._new_empty_row()
                rejected = []
                for line_dict in lines:
                    if self.has_thematic_split():
                        thematic = line_dict['annotation']
                        prodigy_label_map = self.get_prodigy_label_map(
                            thematic)
                    else:
                        prodigy_label_map = self.get_prodigy_label_map()
                    # get class labels:
                    class_ids = line_dict['accept']
                    new_row['text'] = line_dict['text']
                    new_row['id'] = line_dict['record_id']

                    if line_dict['answer'] == 'reject':
                        rejected.append(line_dict['record_id'])
                    else:
                        for class_id in class_ids:
                            # set the class label to 1
                            class_label = prodigy_label_map[class_id]
                            new_row[class_label] = 1

                # check if the same abstract was accepted and rejected -> annotation error
                if self._thematic_split and (len(rejected) == 1 or len(rejected) == 2):
                    raise ValueError(
                        f'Same abstract was accepted and rejected {rejected[0]}')

                if rejected:
                    self.rejected.append(rejected[0])
                    self.nr_rejected += 1
                else:
                    all_rows.append(new_row)
                # reset for next abstract
                lines = []
                rejected = []
        self.df = pd.concat([self.df, pd.DataFrame(all_rows)])
        # reorder rows according to the id
        self.df = self.df.sort_values(by='id')

    def _read_all_ner(self) -> None:
        with open(self.jsonl_path, 'r', encoding='utf-8') as infile:
            has_ner = False
            for line in infile:
                line_dict = json.loads(line)
                if line_dict['answer'] == 'reject':
                    continue
                try:
                    spans = line_dict['spans']
                    has_ner = True
                except KeyError:
                    continue

                record_id = line_dict['tokens']
                id = line_dict['record_id']
                # check if id already exists
                for span in spans:
                    if 'label' in span.keys():
                        start = span['token_start']
                        end = span['token_end']
                        label = span['label']
                        self.ner_tags.add(label)
                        self.span_labels.add(label)
                        ner_tokens = [record_id[i]['text']
                                      for i in range(start, end+1)]
                        self.ner_per_abstract[id][label].append(
                            ner_tokens)

            if not has_ner:
                print(f'No NER data found in {self.jsonl_path}')

    def _new_empty_row(self) -> dict:
        column_names = list(self.df.columns)
        return {col: 0 for col in column_names}

    def _check_order(self) -> None:
        ordered_file = self.jsonl_path.replace('.jsonl', '_reordered.jsonl')
        records = {}
        with open(self.jsonl_path, 'r', encoding='utf-8') as infile:
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
        with open(ordered_file, 'w', encoding='utf-8') as outfile:
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

    @staticmethod
    def replace_newlines(df):
        return df.map(lambda x: x.replace('\n', '\\n') if isinstance(x, str) else x)


class ProdigyDataCollector():
    def __init__(self, list_of_files: list[str], annotators: list[str], expert_annotator='', purposes: list[str] = None) -> None:
        self.expert_annotator = expert_annotator
        self.prodigy_files = list_of_files
        self.prodigy_readers = []
        if not purposes:
            purposes = ['both'] * len(list_of_files)
        for file, name, purpose in zip(list_of_files, annotators, purposes):
            prodigy_reader = ProdigyDataReader(file, name, purpose)
            self.prodigy_readers.append(prodigy_reader)

        for reader in self.prodigy_readers:
            reader.df['annotator'] = reader.annotator
        self.tasks = {}
        self.duplicates = []

        self._check_tasks([reader for reader in self.prodigy_readers if reader.purpose == 'class'] + [reader for reader in self.prodigy_readers if reader.purpose == 'both'])
        self._read_all()

        self._check_duplicates()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, record_id: str) -> pd.DataFrame:
        return self.df[self.df['id'] == record_id]

    def _read_all(self) -> None:  
        self.df = pd.concat([reader.df for reader in self.prodigy_readers if reader.purpose == 'class' or reader.purpose == 'both'])
        # merge all ner_per_abstract dictionaries
        self.ner_per_abstract = {}
        self.span_labels = set()
        for reader in self.prodigy_readers:
            if reader.purpose == 'ner' or reader.purpose == 'both':
                self.ner_per_abstract.update(reader.ner_per_abstract)
                self.span_labels = self.span_labels.union(reader.span_labels)

    def _check_tasks(self, readers: list[ProdigyDataReader]) -> None:
        # check if all tasks are the same
        tasks_first = readers[0].get_classification_tasks()
        for reader in readers[1:]:
            tasks = reader.get_classification_tasks()
            if tasks != tasks_first:
                print(tasks)
                print(tasks_first)
                print(self.compare_dictionaries(tasks, tasks_first))
                raise ValueError(
                    f'Tasks are not the same in {reader.jsonl_path} and {readers[0].jsonl_path}')
        self.tasks = tasks_first

    @staticmethod
    def compare_dictionaries(dict1, dict2):
        differences = {
            'value_differences': {},
            'key_differences': {}
        }

        # Finding keys in both dictionaries
        all_keys = set(dict1.keys()).union(dict2.keys())

        # Iterate over each key
        for key in all_keys:
            value1 = dict1.get(key, '__MISSING__')
            value2 = dict2.get(key, '__MISSING__')

            if key in dict1 and key in dict2:
                if value1 != value2:
                    differences['value_differences'][key] = (value1, value2)
            else:
                differences['key_differences'][key] = (
                    value1 if key in dict1 else None, value2 if key in dict2 else None)

        return differences

    def _check_duplicates(self) -> None:
        duplicates = self.df[self.df.duplicated(subset='id', keep=False)]
        if not duplicates.empty:
            # check if there is duplicates from the same annotator, by checking annotator column
            duplicates_annotator = self.df[self.df.duplicated(
                subset=['id', 'annotator'])]
            if not duplicates_annotator.empty:
                raise ValueError(
                    f'Same sample has been annotated by same annotator: {duplicates_annotator}')
            else:
                duplicate_ids = list(duplicates['id'].unique())
                if not self.expert_annotator:
                    raise ValueError(
                        f'Duplicates found and no expert annotator specified. Please implement a solution for duplicates')
                else:
                    for id in duplicate_ids:
                        for reader in self.prodigy_readers:
                            if id in reader:
                                if reader.annotator != self.expert_annotator:
                                    reader.remove_ids([id])
                self.duplicates = duplicate_ids
                self._read_all()
        # drop annotator column
        self.df = self.df.drop(columns='annotator')

    @property
    def nr_rejected(self) -> int:
        return sum([reader.nr_rejected for reader in self.prodigy_readers])

    def visualize_dist(self, x_label: str = None, save_path: str = None) -> None:
        if x_label is None:
            classification_tasks = list(self.tasks.keys())

            # Set up the subplots
            num_tasks = len(classification_tasks)
            num_cols = 5
            num_rows = (num_tasks + num_cols - 1) // num_cols
            fig, axes = plt.subplots(
                nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
            axes = axes.flatten()  # Flatten axes if more than 1 row

            for idx, task in enumerate(classification_tasks):
                ax = axes[idx]
                task_freq = self.get_onehot_task_df(
                    task).drop(columns=FIXED_COLUMNS).sum()
                self._plot_task_dist(task_freq, task, ax)

            # Hide any unused subplots
            for idx in range(num_tasks, len(axes)):
                fig.delaxes(axes[idx])

            plt.subplots_adjust(hspace=0.90, top=0.95, bottom=0.15)

            fig.suptitle('Overview of all Classification Tasks', fontsize=16)

        else:
            task_freq = self.get_onehot_task_df(
                x_label).drop(columns=FIXED_COLUMNS).sum()
            self._plot_task_dist(task_freq, x_label)

        if save_path:
            # check if it's file or folder
            if os.path.isdir(save_path):
                save_path = os.path.join(
                    save_path, f'{datetime.now().strftime("%Y%m%d")}_task_dist.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_task_dist(self, frequencies: pd.Series, x_label: str, ax=None):
        if ax is None:
            ax = plt.gca()

        sns.set_theme(style="whitegrid")

        sns.barplot(x=frequencies.index, y=frequencies.values, ax=ax)

        ax.set_ylabel('Count')
        ax.set_xlabel(x_label)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        for p in ax.patches:
            height = p.get_height()
            if height > 5:
                ax.annotate(format(height, '.0f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, -10),
                            textcoords='offset points')
            else:
                ax.annotate(format(height, '.0f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='top',
                            xytext=(0, 12),
                            textcoords='offset points')

    def _is_valid_task(self, task_name: str) -> Union[bool, None]:
        if not task_name in self.tasks.keys():
            raise ValueError(
                f'Invalid task name, options are ´{self.tasks.keys()}´')
        else:
            return True

    def is_multilabel(self, task_name: str) -> bool:
        readers = [reader for reader in self.prodigy_readers if reader.purpose == 'class' or reader.purpose == 'both']
        if self._is_valid_task(task_name):
            for reader in readers:
                # if at least one reader has a multi-label task, return True
                if reader._is_task_multi_label(task_name):
                    return True
            return False

    def get_label_task_df(self, task: str) -> tuple[dict[int, str], pd.DataFrame]:
        readers = [reader for reader in self.prodigy_readers if reader.purpose == 'class' or reader.purpose == 'both']
        if self._is_valid_task(task):
            label_to_int, task_df = readers[0].get_label_task_df(
                task)
            for reader in readers[1:]:
                label_to_int, reader_task_df = reader.get_label_task_df(
                    task, label_to_int)
                task_df = pd.concat([task_df, reader_task_df])

            # TODO: Make more efficient
            if not any(task_df[task].apply(len) > 1):
                task_df[task] = task_df[task].apply(lambda x: x[0])

            return label_to_int, task_df

    def get_onehot_task_df(self, task_name: str) -> pd.DataFrame:
        readers = [reader for reader in self.prodigy_readers if reader.purpose == 'class' or reader.purpose == 'both']
        if self._is_valid_task(task_name):
            task_filtered_df = readers[0].get_onehot_task_df(
                task_name)
            for reader in readers[1:]:
                reader_task_df = reader.get_onehot_task_df(task_name)
                task_filtered_df = pd.concat(
                    [task_filtered_df, reader_task_df])

            return task_filtered_df

    def visualize_nr_dist(self, x_label: str = None, save_path: str = None) -> None:
        if x_label is None:
            classification_tasks = list(self.tasks.keys())

            # Set up the subplots
            num_tasks = len(classification_tasks)
            num_cols = 5
            num_rows = (num_tasks + num_cols - 1) // num_cols
            fig, axes = plt.subplots(
                nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
            axes = axes.flatten()  # Flatten axes if more than 1 row

            for idx, task in enumerate(classification_tasks):
                ax = axes[idx]
                task_freq = self.get_onehot_task_df(
                    task).drop(columns=FIXED_COLUMNS)
                task_freq = task_freq.sum(axis=1).value_counts()
                self._plot_task_dist(task_freq, f'Nr of Labels for {task}', ax)

            # Hide any unused subplots
            for idx in range(num_tasks, len(axes)):
                fig.delaxes(axes[idx])

            plt.subplots_adjust(hspace=0.8, top=0.95, bottom=0.15)

            fig.suptitle(
                'Number of Labels of all Classification Tasks', fontsize=16)

        else:
            task_freq = self.get_onehot_task_df(
                x_label).drop(columns=FIXED_COLUMNS)
            task_freq = task_freq.sum(axis=1).value_counts()
            self._plot_task_dist(task_freq, f'Nr of labels for {x_label}')

        if save_path:
            # check if it's file or folder
            if os.path.isdir(save_path):
                save_path = os.path.join(
                    save_path, f'{datetime.now().strftime("%Y%m%d")}_nr_dist.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def get_ner_per_abstract(self, record_id: int, label: str = None) -> list[(str, str)]:
        ners = []
        if label:
            for ner in self.ner_per_abstract[record_id][label]:
                ners.append((' '.join(ner), label))
        else:
            for label in self.ner_per_abstract[record_id]:
                for ner in self.ner_per_abstract[record_id][label]:
                    ners.append((' '.join(ner), label))
        return ners

    def get_ner_stats(self, save_path: str = None) -> None:
        """
        Report entities frequency of NER (avg. per abstract), 
        entities frequency per label, and average NER length.

        Args:
            save_path (str, optional): Path to save the report. Defaults to None.
        """
        report_lines = []
        if len(self.ner_per_abstract) > 0:
            nr_abstracts = len(self)
            nr_total_entities = 0
            nr_abstracts_without_ner = 0

            for abstract in self.ner_per_abstract.values():
                for entity in abstract.values():
                    if len(entity) == 0:
                        nr_abstracts_without_ner += 1
                    nr_total_entities += len(entity)

            avg_nr_entities_per_abstract = nr_total_entities / nr_abstracts

            avg_ner_per_abstract_per_label = {}
            nr_ner_per_label = {}

            for label in self.span_labels:
                nr_entities = 0
                for abstract in self.ner_per_abstract.values():
                    if label in abstract:
                        nr_entities += len(abstract[label])
                avg_ner_per_abstract_per_label[label] = nr_entities / nr_abstracts
                nr_ner_per_label[label] = nr_entities

            avg_ner_length_per_label = {}

            for label in self.span_labels:
                nr_entities = 0
                total_length = 0
                for abstract in self.ner_per_abstract.values():
                    if label in abstract:
                        for entity in abstract[label]:
                            total_length += len(entity)
                            nr_entities += 1
                avg_ner_length_per_label[label] = total_length / nr_entities

            # Prepare the report
            report_lines.append('Total number of entities per label:')
            for label, nr in nr_ner_per_label.items():
                report_lines.append(f'\t {label}: {nr}')

            report_lines.append(f'Average number of entities per abstract: {
                                avg_nr_entities_per_abstract}')

            report_lines.append(
                'Average number of entities per abstract per label:')
            for label, avg in avg_ner_per_abstract_per_label.items():
                report_lines.append(f'\t {label}: {avg}')

            report_lines.append('Average length of entities per label:')
            for label, avg in avg_ner_length_per_label.items():
                report_lines.append(f'\t {label}: {avg}')
            
            report_lines.append(f'Number of abstracts without NER: {nr_abstracts_without_ner}')
        else:
            report_lines.append('No NER data found in the dataset')
        # Save to file or print
        if save_path:
            if os.path.isdir(save_path):
                save_path = os.path.join(
                    save_path, f'{datetime.now().strftime("%Y%m%d")}_ner_stats.txt')
            with open(save_path, 'w', encoding='utf-8') as file:
                file.write('\n'.join(report_lines))
        else:
            for line in report_lines:
                print(line)

    def get_nr_annot_stats(self, save_path: str = None) -> None:
        nr_rejected = 0
        for reader in self.prodigy_readers:
            nr_rejected += reader.nr_rejected
        nr_total = sum([reader.nr_total for reader in self.prodigy_readers])

        # print or save the number of annotations and rejected samples in total
        report_lines = []
        report_lines.append(
            f'Total number of annotations: {nr_total}')
        report_lines.append(f'Total number of rejected samples: {nr_rejected}')
        report_lines.append(
            f'Total number of valid annotations:  {len(self.df)}')
        report_lines.append(f'Total number of duplicates: {
                            len(self.duplicates)}')
        report_lines.append(f'Duplicates: {self.duplicates}')
        if save_path:
            # check if it's file or folder
            if os.path.isdir(save_path):
                save_path = os.path.join(
                    save_path, f'{datetime.now().strftime("%Y%m%d")}_nr_stats.txt')
            with open(save_path, 'w', encoding='utf-8') as file:
                file.write('\n'.join(report_lines))
        else:
            for line in report_lines:
                print(line)


class ProdigyIAAHelper():
    def __init__(self, list_of_files: list[str], names: list[str] = None, log: str = 'iaa.log') -> None:
        self.prodigy_files = list_of_files
        self.prodigy_readers = []
        self.log = log
        self.names = names if names else [
            f'Annotator_{i}' for i in range(len(list_of_files))]

        for file, name in zip(list_of_files, names):
            prodigy_reader = ProdigyDataReader(file, name)
            self.prodigy_readers.append(prodigy_reader)

        self._inital_log()
        self.tasks = {}
        self.iaa_dir = None
        self.iaa_file = None
        # sanity checks
        self._inspect_rejected()
        self._check_number_of_samples()
        self._check_order_of_samples()
        self._check_tasks()
        print('Sanity checks passed')

    def _inital_log(self):
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log, 'w', encoding='utf-8') as f:
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
        with open(self.log, 'a', encoding='utf-8') as f:
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

    def agreement_per_task(self, task: str, readers: list[ProdigyDataReader] = None, save: bool = False, measures: list[str] = None) -> dict[str, Union[float, str]]:
        iaa_dict = {meas: None for meas in measures}

        if self._is_valid_task(task):
            int_to_label, task_df = self.get_task_df(task)
            readers = self.prodigy_readers if readers is None else readers
            if 'Krippendorff' in measures:

                alpha, low, high = calculate_krippendorff_alpha_with_ci(
                    task_df, 'id', 'reader', task)
                confidence_interval = high - low

                # if only two annotators, plot confusion matrix
                if len(readers) == 2:
                    labels = list(int_to_label.keys())
                    first_pred = task_df[task_df['reader'] == self.names[0]].sort_values(by='id')[
                        task]
                    second_pred = task_df[task_df['reader'] == self.names[1]].sort_values(by='id')[
                        task]
                    first_pred_matrix = MultiLabelBinarizer(
                        classes=labels).fit_transform(first_pred.to_numpy())
                    second_pred_matrix = MultiLabelBinarizer(
                        classes=labels).fit_transform(second_pred.to_numpy())
                    cm = multilabel_confusion_matrix(
                        first_pred_matrix, second_pred_matrix, labels=labels)
                    iaa_str = f'Krippendorff\'s Alpha: {round(alpha, 2)}, CI: {
                        round(confidence_interval, 2)}'
                    plot = self.plot_multilabel_confusion_matrix(
                        cm, list(int_to_label.values()), task, iaa_str)

                    if save:
                        annotators = '_'.join(self.names)
                        task_name = task.replace(' ', '_').lower()
                        plt.savefig(
                            f'{self.iaa_dir}/{self.iaa_file}_{annotators}_{task_name}.png', bbox_inches='tight', dpi=300)
                        plt.close()
                    else:
                        plot.show()

                iaa_dict['Krippendorff'] = alpha
                iaa_dict['Krippendorff_Quality'] = interpret_alpha(alpha)
                iaa_dict['Krippendorff_CI'] = confidence_interval

            if 'Cohen' in measures:
                if not self._task_is_multi_label(task):

                    # If only two annotators, plot confusion matrix
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

                        # Check if there are nan values, replace them with -1
                        if reader1_task_list.isnull().values.any() or reader2_task_list.isnull().values.any():
                            reader1_task_list.fillna(-1, inplace=True)
                            reader2_task_list.fillna(-1, inplace=True)
                            int_to_label[-1] = 'None'

                        cm = confusion_matrix(
                            reader1_task_list, reader2_task_list, labels=list(int_to_label.keys()))
                        measure = "Cohen's Kappa"
                        cohen, ci = calculate_cohen_kappa_from_cfm_with_ci(cm)

                        plot = self.plot_confusion_matrix(
                            cm, list(int_to_label.values()), task, f'{measure}: {round(cohen, 2)}, CI: {round(ci, 2)}')

                        if save:
                            annotators = '_'.join(
                                [reader.annotator for reader in readers])
                            task_name = task.replace(' ', '_').lower()
                            plt.savefig(
                                f'{self.iaa_dir}/{self.iaa_file}_{annotators}_{task_name}.png', bbox_inches='tight', dpi=300)
                            plot.close()
                        else:
                            plot.show()

                        iaa_dict['Cohen'] = cohen
                        iaa_dict['Cohen_Quality'] = interpret_kappa(cohen)
                        iaa_dict['Cohen_CI'] = ci

                    else:
                        pair_list = list(combinations(self.prodigy_readers, 2))
                        average_kappa = 0
                        average_boundary_limits = 0

                        for c in pair_list:
                            partial_iaa_dict = self.agreement_per_task(
                                task, c, measures=['Cohen'], save=True)
                            kappa = partial_iaa_dict['Cohen']
                            ci = partial_iaa_dict['Cohen_CI']
                            average_kappa += kappa
                            average_boundary_limits += ci

                        average_kappa /= len(pair_list)
                        average_boundary_limits /= len(pair_list)

                        iaa_dict['Cohen'] = average_kappa
                        iaa_dict['Cohen_Quality'] = interpret_kappa(
                            average_kappa)
                        iaa_dict['Cohen_CI'] = average_boundary_limits

                else:
                    print(
                        f'{task} is a multi-label task, skipping Cohen\'s Kappa calculation')

            if 'Percent_Agreement' in measures:
                agreement = calculate_percentage_agreement(
                    task_df, 'id', 'reader', task)
                iaa_dict['Percent_Agreement'] = agreement

            return iaa_dict

    def agreement_all_tasks(self, pprint: bool = True, csv_path: str = None, save: bool = True, measures: list[str] = None) -> None:
        # Get directory of the csv_path
        self.iaa_dir = os.path.dirname(csv_path)
        self.iaa_file = os.path.basename(csv_path).split('.')[0]
        if not measures:
            measures = ['Krippendorff', 'Cohen', 'Percent_Agreement']

        agreement_data = []
        for task in self.tasks.keys():
            iaa_dict = self.agreement_per_task(
                task, save=save, measures=measures)
            iaa_dict['Task'] = task
            agreement_data.append(iaa_dict)
            if pprint:
                for meas in measures:
                    value = iaa_dict[meas]
                    try:
                        quality = iaa_dict[f'{meas}_Quality']
                        ci_boundary = iaa_dict[f'{meas}_CI']
                        print(
                            f'{task} - {meas}: {value}, CI boundary: {ci_boundary} --> {quality}')
                    except KeyError:
                        print(f'{task} - {meas}: {value}')

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
                _, task_dfs = reader.get_label_task_df(task_name)
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

    def plot_confusion_matrix(self, cm, labels, task_name: str, iaa_str: str = None):
        dist = ConfusionMatrixDisplay(
            cm, display_labels=labels)
        plt.figure(figsize=(10, 8))
        dist.plot(xticks_rotation='vertical', ax=plt.gca())
        plt.xlabel(self.names[0].capitalize(), fontsize=16)
        plt.ylabel(self.names[1].capitalize(), fontsize=16)

        plt.title(f'{task_name} - Single Label', fontsize=16, pad=20)
        plt.tight_layout()
        if iaa_str:
            title_height = plt.gca().title.get_position()[1]
            text_y = title_height + 0.02
            plt.text(0.5, text_y, iaa_str, ha='center', va='center',
                     fontsize=12, transform=plt.gca().transAxes)
        return plt

    def plot_multilabel_confusion_matrix(self, cm, labels, task_name: str = None, iaa_str: str = None) -> Figure:
        """
        Plots the multilabel confusion matrix.

        Parameters:
        cm (ndarray): The multilabel confusion matrix.
        labels (list): The list of labels.
        """
        n_labels = len(labels)
        n_cols = 3
        n_rows = (n_labels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for i, (ax, label) in enumerate(zip(axes, labels)):
            tn, fp, fn, tp = cm[i].ravel()
            matrix = np.array([[tn, fp], [fn, tp]])

            cax = ax.matshow(matrix, cmap=plt.cm.Blues)
            fig.colorbar(cax, ax=ax, shrink=0.75)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Negative', 'Positive'])
            ax.set_yticklabels(['Negative', 'Positive'])
            ax.set_title(f'{label}')

            ax.set_xlabel(self.names[1].capitalize())
            ax.set_ylabel(self.names[0].capitalize())
            for (j, k), val in np.ndenumerate(matrix):
                ax.text(k, j, f'{val}', ha='center', va='center')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        if task_name:
            fig.suptitle(f'{task_name} - Multi Label', fontsize=16)

        fig.subplots_adjust(top=0.94, wspace=0.5, hspace=0.1)

        if iaa_str:
            fig.text(0.5, 0.95, iaa_str, ha='center', va='top', fontsize=12)

        return fig
