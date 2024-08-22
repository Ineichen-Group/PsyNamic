import os
from abc import abstractmethod
from os import path
from typing import Iterable, Union, Optional

import json
import numpy as np
import pandas as pd
import torch
import csv
import ast
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from torch.utils.data import Dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from transformers import AutoTokenizer
from datetime import datetime

# Fix the random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

MAX_MODEL_LENGTH = {
    'bert': 512,
}

# To avoid circular import has to live here
MODEL_IDENTIFIER = {
    'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    'biomedbert-abstract': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
    'scibert': 'allenai/scibert_scivocab_uncased',
    'biobert': 'dmis-lab/biobert-v1.1',
    'clinicalbert': 'emilyalsentzer/Bio_ClinicalBERT',
    'biolinkbert': 'michiyasunaga/BioLinkBERT-base',
}

# TODO: Fix use_val vs. self.val vs. parameter use_val mess
# TODO: Use some inheritence with DataHandler and DataHandlerBIO
# TODO: Make load and get more consistent
# TODO: Typing and docstrings

class DataSplit(Dataset):
    "PyTorch Dataset class for a given data split."
    ID_COL = 'id'
    TEXT_COL = 'text'
    LABEL_COL = 'labels'

    def __init__(self, split: pd.DataFrame, id2label: dict[int, str], tokenizer, max_len: str, multilabel: bool) -> None:
        self.df = split
        self.is_multilabel = multilabel
        self.max_len = max_len
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.tokenizer = tokenizer
        self._index = 0  # index for iteration

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        text = self.df.iloc[idx][self.TEXT_COL]
        labels = self.df.iloc[idx][self.LABEL_COL]

        if self.is_multilabel:
            labels = ast.literal_eval(labels)
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = torch.tensor(labels, dtype=torch.long)

        try:
            # TODO: save data encoded to save time
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,  # TODO: Check if max length is correct
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )
        except ValueError:
            breakpoint()

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

    def __eq__(self, other) -> bool:
        if not isinstance(other, DataSplit):
            return False
        return self.df.equals(other.df)

    def __repr__(self) -> str:
        return f"DataSplit(num_samples={len(self.df)}, labels={self.labels})"

    def __iter__(self):
        self._index = 0  # Reset index at the start of iteration
        return self

    def __next__(self):
        if self._index < len(self.df):
            id_ = self.df.iloc[self._index][self.ID_COL]
            text = self.df.iloc[self._index][self.TEXT_COL]
            labels = self.df.iloc[self._index][self.LABEL_COL]
            self._index += 1
            return id_, text, labels
        else:
            raise StopIteration

    def to_csv(self, save_path: str) -> None:
        self.df.to_csv(save_path, index=False)

    @property
    def labels(self) -> list[int]:
        if self.is_multilabel:
            # get first row of labels
            labels = self.df[self.LABEL_COL].iloc[0]
            labels = ast.literal_eval(labels)
            return labels

        else:
            label_list = self.df[self.LABEL_COL].unique().tolist()
            label_list.sort()
            return label_list

    @property
    def nr_labels(self) -> int:
        return len(self.labels)

    def overlap(self, others: list['DataSplit']) -> bool:
        """Check if the IDs of the current DataSplit overlap with the IDs of other DataSplits."""
        current_ids = set(self.df[self.ID_COL])
        for other in others:
            other_ids = set(other.df[self.ID_COL])
            if current_ids.intersection(other_ids):
                return True
        return False


class DataSplitBIO(DataSplit):
    """ PyTorch Dataset class for a given data split for NER tasks.

    """
    TOKEN_COL = 'tokens'
    NER_COL = 'ner_tags'

    def __init__(self, split: pd.DataFrame, label2id: dict, tokenizer, max_len: int) -> None:
        self.df = split
        self.max_len = max_len
        self.label2id = label2id
        self.tokenizer = tokenizer

        # Convert string representations of lists to lists
        self.df[self.TOKEN_COL] = self.df[self.TOKEN_COL].apply(
            ast.literal_eval)
        self.df[self.NER_COL] = self.df[self.NER_COL].apply(ast.literal_eval)

        # Make sure the ids of label2id are integers
        self.label2id = {k: int(v) for k, v in self.label2id.items()}

    def __getitem__(self, idx: int) -> dict:
        tokens = self.df.iloc[idx][self.TOKEN_COL]
        ner_tags = self.df.iloc[idx][self.NER_COL]

        # Tokenize the tokens
        encoding = self.tokenizer(
            tokens,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            is_split_into_words=True,
            return_tensors='pt'
        )
        # print tokens created by tokenizer
        bert_tokens = self.tokenizer.convert_ids_to_tokens(
            encoding['input_ids'].squeeze(0))

        # Align labels with BERT tokens
        labels = [self.label2id[tag] for tag in ner_tags]
        label_ids = [-100] * self.max_len

        # Each token is assigned an id: if token is split into multiple subtokens, all subtokens get the same id
        word_ids = encoding.word_ids(batch_index=0)
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                # -100 as a dummy label for padding tokens or ignored subtokens
                label_ids[i] = -100
            elif word_idx != previous_word_idx:  # Label only the first subtoken to avoid redundant labels for word pieces
                label_ids[i] = labels[word_idx]
            previous_word_idx = word_idx

        # Add the labels to the encoding
        encoding["labels"] = torch.tensor(label_ids)

        # Convert tensor dimensions from (1, max_len) to (max_len)
        return {key: val.squeeze(0) for key, val in encoding.items()}

    @property
    def labels(self) -> list[str]:
        return list(self.label2id.keys())


class DataHandler():
    """ Abstract DataHandler class to handle data loading, preprocessing and splitting.
        Idea: Inherit from this class and implement the abstract methods to create a DataHandler for a specific dataset.
    """
    ID_COL = 'id'
    TEXT_COL = 'text'
    LABEL_COL = 'labels'
    ANNOTATOR_COL = 'annotator'

    def __init__(self, model: str = 'scibert', data_path: str = None, meta_file: str = None, int_to_label: str = None, ) -> None:
        self.model = MODEL_IDENTIFIER[model]
        self.nr_classes = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        # Provide either meta_file or int_to_label
        if not meta_file and not int_to_label:
            raise ValueError(
                'Provide either a meta_file or int_to_label dictionary.')

        if meta_file:
            meta_data = json.load(open(meta_file))
            filename = path.basename(data_path)
            self.is_multilabel = meta_data['Is_multilabel']
            if "Int_to_label" in meta_data:
                self.id2label = meta_data['Int_to_label']
                self.nr_classes = len(self.id2label)

        elif int_to_label:
            self.id2label = int_to_label
            self.df = self.read_in_data(data_path)
            self.is_multilabel = self._check_if_multilabel()
            self.nr_classes = self._determine_nr_classes()

        # Case where data is provided unsplitted
        # (if it's a directory instead, it's assumed that the splits are already saved in the directory)
        if path.isfile(data_path):
            self.df = self.read_in_data(data_path)
            if not self.nr_classes:
                self.nr_classes = self._determine_nr_classes()
            self.max_len = self._detect_length()

        self.train = None
        self.test = None
        self.val = None
        self.folds = []

        # Splits data
        self.train_size = 0.8
        self.n_splits = 5
        self.use_val = False

    def __len__(self) -> int:
        return len(self.df)

    def _detect_length(self) -> int:
        """
        Detect a suitable max_length based on token lengths in the dataset.

        Args:
            percentile (float): Percentile to determine max_length.

        Returns:
            int: Suggested max_length for tokenization.
        """
        percentile = 95
        # Tokenize all texts in the dataset and measure their lengths
        lengths = []
        for text in self.df[self.TEXT_COL]:
            encoded = self.tokenizer.encode(text, add_special_tokens=True)
            lengths.append(len(encoded))

        if 'bert' in self.model.lower():
            max_model_length = MAX_MODEL_LENGTH['bert']
            max_length = int(np.percentile(lengths, percentile))
            if max_length > max_model_length:
                max_length = max_model_length
        return max_length

    def _determine_nr_classes(self) -> int:
        """ 
        Determine the number of classes in the dataset. In case of multilabel classification, return the number of labels and not combinations of labels.
        """
        if self.is_multilabel:
            return len(self.df[self.LABEL_COL].iloc[0])
        else:
            return len(self.df[self.LABEL_COL].unique())

    def _check_if_multilabel(self) -> bool:
        return any(isinstance(label, list) for label in self.df[self.LABEL_COL])

    def _check_presence_of_labels(self, datasplit: pd.DataFrame) -> bool:
        """ Check if all labels are present in a given datasplit; used for sanity checkking during splitting."""
        if datasplit is None:
            return True
        return self.nr_classes == len(datasplit[self.LABEL_COL].unique())

    def get_strat_split(self, train_size: float = 0.8, use_val: bool = False, seed: int = SEED) -> tuple[DataSplit, DataSplit, Union[DataSplit, None]]:
        """ Get stratified split of the data, with an optional validation set.

        Args:
            train_size (float, optional): Size of the training set. Defaults to 0.8.
            use_val (bool, optional): Whether to use a validation set. Defaults to False.
            seed (int, optional): Random seed. Defaults to SEED.

        Returns:
            tuple[DataSplit, DataSplit, Union[DataSplit, None]]: Train, test and validation set. Validation set is None if use_val is False.
        """
        def reuse():
            if self.train is None:
                return False
            # Check if the split with the same parameters has already been created, if so, return the split (saves compute time)
            elif self.train is not None and self.test is not None and self.use_val == use_val and self.train_size == train_size:
                return True
            else:
                return False
        if reuse():
            # check if there is overlap between the splits
            train = DataSplit(self.train, self.id2label,
                              self.tokenizer, self.max_len, self.is_multilabel)
            test = DataSplit(self.test, self.id2label,
                             self.tokenizer, self.max_len, self.is_multilabel)
            val = DataSplit(self.val, self.id2label, self.tokenizer,
                            self.max_len, self.is_multilabel)
            if use_val:
                if train.overlap([test, val]):
                    raise ValueError('Overlap between splits detected.')
                return train, test, val
            else:
                if train.overlap([test]):
                    raise ValueError('Overlap between splits detected.')
                return train, test, None

        else:
            self.train_size = train_size
            self.use_val = use_val

        if self.is_multilabel:
            # convert df into np array
            X = self.df[self.TEXT_COL].values.tolist()
            y = self.df[self.LABEL_COL].values.tolist()
            if use_val:
                # First split: train (e.g. 60%) and remaining (e.g. 40%)
                msss1 = MultilabelStratifiedShuffleSplit(
                    n_splits=1, test_size=1-train_size, random_state=SEED)
                train_index, remaining_index = next(msss1.split(X, y))
                train_df = self.df.iloc[train_index]
                remaining_df = self.df.iloc[remaining_index]

                # Second split: remaining (40%) into val (20%) and test (20%)
                msss2 = MultilabelStratifiedShuffleSplit(
                    n_splits=1, test_size=0.5, random_state=SEED)
                val_index, test_index = next(msss2.split(
                    remaining_df[self.TEXT_COL].values, np.array(remaining_df[self.LABEL_COL].tolist())))
                val_df = remaining_df.iloc[val_index]
                test_df = remaining_df.iloc[test_index]

                # Save splits to avoid recomputation
                self.train = train_df
                self.val = val_df
                self.test = test_df
            else:
                msss = MultilabelStratifiedShuffleSplit(
                    n_splits=1, test_size=1-train_size, random_state=SEED)
                train_index, test_index = next(msss.split(X, y))
                train_df = self.df.iloc[train_index]
                test_df = self.df.iloc[test_index]
                # Save splits to avoid recomputation
                self.train = train_df
                self.test = test_df
                self.val = None
        else:
            if use_val:
                # First split: train (e.g. 60%) and remaining (e.g. 40%)
                split = StratifiedShuffleSplit(
                    n_splits=1, train_size=train_size, test_size=1-train_size, random_state=SEED)
                train_idx, remaining_idx = next(
                    split.split(self.df, self.df[self.LABEL_COL]))
                train_df = self.df.iloc[train_idx]
                remaining_df = self.df.iloc[remaining_idx]

                # Second split: remaining (e.g. 40%) into val (20%) and test (e.g. 20%)
                val_test_split = StratifiedShuffleSplit(
                    n_splits=1, train_size=0.5, test_size=0.5, random_state=SEED)
                val_idx, val_idx = next(val_test_split.split(
                    remaining_df, remaining_df[self.LABEL_COL]))
                val_df = remaining_df.iloc[val_idx]
                test_df = remaining_df.iloc[val_idx]
                self.train = train_df
                self.val = val_df
                self.test = test_df
            else:
                # Direct split: train (e.g. 80%) and test (e.g. 20%)
                split = StratifiedShuffleSplit(
                    n_splits=1, train_size=train_size, random_state=SEED)
                train_idx, val_idx = next(split.split(
                    self.df, self.df[self.LABEL_COL]))
                train_df = self.df.iloc[train_idx]
                test_df = self.df.iloc[val_idx]
                self.train = train_df
                self.test = test_df

                if not self._check_presence_of_labels(test_df):
                    raise ValueError(
                        'Not all labels are present in the test set; data set might be too small or there is an error in the code.')

            if not (self._check_presence_of_labels(train_df) and self._check_presence_of_labels(test_df)):
                raise ValueError(
                    'Not all labels are present in the train and test set; data set might be too small or there is an error in the code.')

        train = DataSplit(train_df, self.id2label, self.tokenizer,
                          self.max_len, self.is_multilabel)
        test = DataSplit(test_df, self.id2label, self.tokenizer,
                         self.max_len, self.is_multilabel)

        if use_val:
            val = DataSplit(val_df, self.id2label, self.tokenizer,
                            self.max_len, self.is_multilabel)
            if train.overlap([test, val]):
                raise ValueError('Overlap between splits detected.')
            return train, test, val
        else:
            if train.overlap([test]):
                raise ValueError('Overlap between splits detected.')
            return train, test, None

    def get_strat_k_fold_split(self, train_size: float = 0.8, n_splits: int = 5, seed: int = SEED) -> tuple[Iterable[tuple[DataSplit, DataSplit]], DataSplit]:
        """Get stratified k-fold split of the data; i.e., n splits into train and validation set, with a test set.

        Args:
            train_size (float, optional): Size of the training set. Defaults to 0.8.
            n_splits (int, optional): Number of splits. Defaults to 5.
            seed (int, optional): Random seed. Defaults to SEED.

        Returns:
            tuple[Iterable[tuple[DataSplit, DataSplit]], DataSplit]: Iterable of k-folds and test split.
        """
        # Obtain the stratified split for training and testing
        train_split, test_split, _ = self.get_strat_split(
            train_size=train_size)

        # Prepare data for k-fold splitting
        X = train_split.df[self.TEXT_COL].values
        y = np.array(train_split.df[self.LABEL_COL].tolist())

        # Initialize k-fold stratifiers
        if self.is_multilabel:
            kfold = MultilabelStratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            kfold = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed)

        # Create fold indices
        fold_indices = list(kfold.split(X, y))

        # Create DataSplit instances for each fold
        folds = []
        for train_idx, val_idx in fold_indices:
            train_df = train_split.df.iloc[train_idx]
            val_df = train_split.df.iloc[val_idx]

            # Create DataSplit instances for this fold
            train_data_split = DataSplit(
                train_df, self.id2label, self.tokenizer, self.max_len, self.is_multilabel)
            val_data_split = DataSplit(
                val_df, self.id2label, self.tokenizer, self.max_len, self.is_multilabel)

            folds.append((train_data_split, val_data_split))

        self.folds = folds
        return folds, test_split

    def save_split(self, save_path: str) -> None:
        """ Save the train, test and validation splits to a given path as csv files.
        """
        if not path.exists(save_path):
            os.makedirs(save_path)
        date = pd.Timestamp.now().strftime("%Y%m%d")
        meta_data = {
            "Task": self.__class__.__name__,
            "Date": date,
            "Int_to_label": self.id2label,
            "Train_size": self.train_size,
            "Use_val": self.use_val,
            "Is_multilabel": self.is_multilabel,
            "Train_size": len(self.train),
            "Test_size": len(self.test),
        }
        if self.use_val:
            meta_data["Val_size"] = len(self.val)

        if self.folds:
            for i, fold in enumerate(self.folds):
                fold[0].to_csv(f'{save_path}/train_fold_{i}.csv',
                               index=False, quoting=csv.QUOTE_ALL)
                fold[1].to_csv(f'{save_path}/test_fold_{i}.csv',
                               index=False, quoting=csv.QUOTE_ALL)
                try:
                    fold[2].to_csv(
                        f'{save_path}/val_fold_{i}.csv', index=False)
                except AttributeError:
                    pass
            meta_data['N_folds'] = len(self.folds)
        else:
            self.train.to_csv(path.join(save_path, 'train.csv'),
                              index=False, quoting=csv.QUOTE_ALL)
            self.test.to_csv(path.join(save_path, 'test.csv'),
                             index=False, quoting=csv.QUOTE_ALL)
            try:
                self.val.to_csv(path.join(save_path, 'val.csv'),
                                index=False, quoting=csv.QUOTE_ALL)
            except AttributeError:
                pass

        meta_file = path.join(save_path, f'meta.json')

        with open(meta_file, 'w') as f:
            json.dump(meta_data, f, indent=4, ensure_ascii=False)

    def load_splits(self, load_path: str) -> bool:
        """ Load train, test and validation splits from a given path.
            To get the usable splits, call get_strat_split() or get_strat_k_fold_split() after loading the splits.
        """
        if not path.exists(load_path):
            raise FileNotFoundError(f'Path {load_path} does not exist.')
        train_path = path.join(load_path, 'train.csv')
        test_path = path.join(load_path, 'test.csv')
        val_path = path.join(load_path, 'val.csv')

        # Check if normal split or k-fold split
        if path.exists(train_path) and path.exists(test_path):
            self.train = pd.read_csv(train_path)
            self.test = pd.read_csv(test_path)
            if path.exists(val_path):
                self.val = pd.read_csv(val_path)
                self.df = pd.concat([self.train, self.test, self.val])
                self.use_val = True
            else:
                self.df = pd.concat([self.train, self.test])
            self.max_len = self._detect_length()
        else:
            files_in_directory = [f for f in os.listdir(
                load_path) if path.isfile(path.join(load_path, f))]

            # Check if there are any files ending with train/test/val and containing fold
            if any('train' in file and 'fold' in file for file in files_in_directory):
                # get largest fold number
                fold_nr = max([int(file.split('_')[-1].split('.')[0])
                               for file in files_in_directory if 'train' in file])
                for i in range(0, fold_nr+1):
                    train_path = path.join(load_path, f'train_fold_{i}.csv')
                    test_path = path.join(load_path, f'test_fold_{i}.csv')
                    train_data = pd.read_csv(train_path)
                    test_data = pd.read_csv(test_path)

                    if path.exists(path.join(load_path, f'val_fold_{i}.csv')):
                        self.use_val = True
                        val_file = path.join(load_path, f'val_fold_{i}.csv')
                        val_data = pd.read_csv(val_file)
                        if i == 0:
                            self.df = pd.concat(
                                [train_data, test_data, val_data])
                    else:
                        if i == 0:
                            self.df = pd.concat([train_data, test_data])
                        val_data = None
                    self.folds.append((train_data, test_data, val_data))
            else:
                raise FileNotFoundError(
                    f'No train/test/val files found in {load_path}.')

        return self.use_val

    def count_label(self, label: str) -> int:
        """ Count the number of occurences of a label in the dataset."""
        if self.is_multilabel:
            return self.df[self.LABEL_COL].apply(lambda x: x[label] == 1).sum()
        else:
            return self.df[self.LABEL_COL].apply(lambda x: x == label).sum()

    def print_label_dist(self) -> None:
        """ Print the distribution of labels in the dataset using id2label."""
        for id, label in self.id2label.items():
            count = self.count_label(int(id))
            print(f'{label}: {count}')

    @abstractmethod
    def read_in_data(self, data_path: str) -> pd.DataFrame:
        """ Read in the data from a given path and return a pandas DataFrame, with columns 'id', 'text' and 'labels'. 
            In case of multilabel classification, 'labels' should be a list of one-hot encoded labels:  
            e.g.    id                      2439
                    text           "I am a text"
                    labels             [0, 1, 0]

            In case of single label classification, 'labels' should be an integer:
            e.g.    id                      2439
                    text           "I am a text"
                    labels                     2

            """
        pass

    @property
    def labels(self) -> list[int]:
        """ Return the unique labels in the dataset."""
        if self.is_multilabel:
            label_tuples = self.df['labels'].apply(tuple)
            unique_labels = set(label_tuples)
            return list(unique_labels)
        else:
            label_list = self.df[self.LABEL_COL].unique().tolist()
            label_list.sort()
            return label_list


class DataHandlerBIO():
    TOKEN_COL = 'tokens'
    NER_COL = 'ner_tags'
    ID_COL = 'id'

    def __init__(self, data_path: str, model: str = 'scibert') -> None:
        self.model = MODEL_IDENTIFIER[model]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

        # if data_path is a file: read in the data
        if path.isfile(data_path):
            self.df = pd.read_json(data_path, lines=True)
            self.label2id = self._detect_labels()
            self.id2label = {v: k for k, v in self.label2id.items()}
            self.max_len = self._detect_length()

        self.train = None
        self.test = None
        self.val = None
        self.use_val = None
        self.train_size = None

    def _detect_labels(self) -> dict:
        """Detect the labels in the dataset and return a dictionary mapping the labels to integers."""
        labels = set()
        for tags in self.df[self.NER_COL]:
            labels.update(tags)
        labels = sorted(list(labels), reverse=True)
        return {label: i for i, label in enumerate(labels)}

    def _detect_length(self) -> int:
        """
        Determine a suitable max_len for tokenized inputs based on the token lengths in the dataframe using a tokenizer.
        """
        percentile = 95

        def tokenize_text(tokens):
            text = " ".join(tokens)  # Reconstruct the text from tokens
            encoded = self.tokenizer.encode(
                text, add_special_tokens=False)  # Tokenize text
            return len(encoded)
        tokenized_texts = self.df[self.TOKEN_COL].tolist()
        token_lengths = [tokenize_text(tokens) for tokens in tokenized_texts]
        max_length = int(np.percentile(token_lengths, percentile))
        if 'bert' in self.model.lower():
            max_model_length = MAX_MODEL_LENGTH['bert']
            if max_length > max_model_length:
                max_length = max_model_length

        return max_length

    def get_split(self, train_size: float = 0.8, use_val: bool = False, seed: int = 42) -> tuple:
        """Get stratified split of the data, with an optional validation set.

        Args:
            train_size (float, optional): Size of the training set. Defaults to 0.8.
            use_val (bool, optional): Whether to use a validation set. Defaults to False.
            seed (int, optional): Random seed. Defaults to 42.

        Returns:
            tuple: Train, test, and validation set. Validation set is None if use_val is False.
        """

        if self.use_val is None:
            self.use_val = use_val
        # If splits have not been created yet, create them
        if self.train is None:
            np.random.seed(seed)

            shuffled_df = self.df.sample(
                frac=1, random_state=seed).reset_index(drop=True)

            # Calculate the split sizes
            n_total = len(shuffled_df)
            if use_val:
                n_train = int(train_size * n_total)
                n_rest = n_total - n_train
                n_val = int(n_rest * 0.5)
                train_df = shuffled_df.iloc[:n_train]
                val_df = shuffled_df.iloc[n_train:n_train + n_val]
                test_df = shuffled_df.iloc[n_train + n_val:]
            else:
                n_train = int(train_size * n_total)
                train_df = shuffled_df.iloc[:n_train]
                test_df = shuffled_df.iloc[n_train:]
                val_df = None

            self.train = train_df
            self.test = test_df
            self.val = val_df
            self.use_val = use_val
            self.train_size = train_size
            self.max_len = self._detect_length()

        # Check for overlaps
        if self.check_overlap():
            raise ValueError('Overlap between splits detected.')
        train = DataSplitBIO(self.train, self.label2id,
                             self.tokenizer, self.max_len)
        test = DataSplitBIO(self.test, self.label2id,
                            self.tokenizer, self.max_len)
        if self.use_val:
            return train, test, DataSplitBIO(self.val, self.label2id, self.tokenizer, self.max_len)
        else:
            return train, test, None

    def save_split(self, directory: str = "data_splits") -> None:
        """Save the train, test, and optionally validation splits to CSV files along with a meta file."""
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        if self.train is None or self.test is None:
            raise ValueError(
                "No splits have been created yet, use get_split() first."
            )

        # Define file paths
        train_path = os.path.join(directory, "train.csv")
        test_path = os.path.join(directory, "test.csv")
        val_path = os.path.join(directory, "val.csv")

        self.train.to_csv(train_path, index=False)
        self.test.to_csv(test_path, index=False)

        if self.val is not None:
            self.val.to_csv(val_path, index=False)

        # Save metadata
        meta = {
            "Task": "NER",
            "Date": datetime.now().strftime("%Y%m%d"),
            "Int_to_label": {v: k for k, v in self.label2id.items()},
            "Train_size": len(self.train),
            "Use_val": self.val is not None,
            "Test_size": len(self.test),
            "Is_multilabel": False,
        }

        meta_path = os.path.join(directory, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        print(f"Splits and metadata saved in directory '{directory}'.")

    def load_splits(self, directory: str) -> None:
        """Load the train, test, and optionally validation splits from a directory and load the meta file."""
        # Load metadata
        with open(os.path.join(directory, "meta.json"), "r") as f:
            meta = json.load(f)

        self.int2label = meta["Int_to_label"]
        self.label2id = {v: k for k, v in self.int2label.items()}
        self.use_val = meta["Use_val"]
        self.train_size = meta["Train_size"]

        # Load splits
        self.train = pd.read_csv(os.path.join(directory, "train.csv"))
        self.test = pd.read_csv(os.path.join(directory, "test.csv"))
        self.val = None
        if self.use_val:
            self.val = pd.read_csv(os.path.join(directory, "val.csv"))

        self.df = pd.concat([self.train, self.test, self.val])
        self.max_len = self._detect_length()

    def check_overlap(self):
        """Check and print overlapping IDs between the train, test, and validation splits."""
        if self.train is None or self.test is None:
            raise ValueError("No splits have been created yet.")

        # Ensure 'id' column exists in the DataFrames
        if 'id' not in self.train.columns or 'id' not in self.test.columns:
            raise ValueError(
                "'id' column is required in train, test, and validation DataFrames.")

        # Initialize overlap report
        overlap_report = {
            "Train-Test": None,
            "Train-Val": None,
            "Test-Val": None
        }

        # Check for overlap between train and test
        overlap_train_test = self.train.merge(self.test, on='id', how='inner')
        if not overlap_train_test.empty:
            overlap_report["Train-Test"] = overlap_train_test['id'].tolist()
            print(
                f"Overlap between Train and Test splits: {overlap_report['Train-Test']}")

        if self.val is not None:
            # Check for overlap between train and validation
            overlap_train_val = self.train.merge(
                self.val, on='id', how='inner')
            if not overlap_train_val.empty:
                overlap_report["Train-Val"] = overlap_train_val['id'].tolist()
                print(
                    f"Overlap between Train and Validation splits: {overlap_report['Train-Val']}")

            # Check for overlap between test and validation
            overlap_test_val = self.test.merge(self.val, on='id', how='inner')
            if not overlap_test_val.empty:
                overlap_report["Test-Val"] = overlap_test_val['id'].tolist()
                print(
                    f"Overlap between Test and Validation splits: {overlap_report['Test-Val']}")


class PsyNamicSingleLabel(DataHandler):

    def __init__(self, data_path: str, relevant_class: str, meta_file: Optional[str] = None) -> None:
        self.relevant_class = relevant_class
        filename = path.basename(data_path)
        task = filename.split('.')[0]
        if not meta_file:
            all_meta_files = [f for f in os.listdir(
                path.dirname(data_path)) if 'meta' in f]
            for file in all_meta_files:
                if task in file:
                    meta_file = path.join(path.dirname(data_path), file)
                    break

        super().__init__(data_path=data_path, meta_file=meta_file)

    def read_in_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        df = df[[self.ID_COL, self.TEXT_COL, self.relevant_class]]
        df.rename(columns={self.relevant_class: self.LABEL_COL}, inplace=True)
        return df


class PsyNamicMultiLabel(DataHandler):

    def __init__(self, data_path: str, meta_file: Optional[str] = None) -> None:
        filename = path.basename(data_path)
        meta_file = data_path.replace('.csv', '_meta.json')
        super().__init__(data_path=data_path, meta_file=meta_file)

    def read_in_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        label_cols = df.columns[df.columns.get_loc(self.ANNOTATOR_COL)+1:]
        self.id2label = {i: col for i, col in enumerate(label_cols)}

        def to_one_hot_encoded(row):
            return row[label_cols].values.tolist()

        # Apply the function to each row and create a new 'labels' column
        df[self.LABEL_COL] = df.apply(to_one_hot_encoded, axis=1)
        df = df[[self.ID_COL, self.TEXT_COL, self.LABEL_COL]]
        return df


class PsychNamicRelevant(DataHandler):
    def __init__(self, data_path: str, id_col: str, title_col: str, abst_col: str, rel_col: str) -> None:
        self.id_col = id_col
        self.title_col = title_col
        self.abst_col = abst_col
        self.rel_col = rel_col
        self.id2label = {0: 'excluded', 1: 'included'}
        super().__init__(data_path=data_path, int_to_label=self.id2label)

    def read_in_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        # Use .copy() to ensure we're working with a copy
        df = df[df[self.rel_col].notna()].copy()
        df[self.rel_col] = df[self.rel_col].astype(int)
        df = df[[self.id_col, self.title_col,
                 self.abst_col, self.rel_col]].copy()
        df.loc[:, self.TEXT_COL] = df[self.title_col] + \
            '.^\n' + df[self.abst_col]
        df.drop(columns=[self.title_col, self.abst_col], inplace=True)
        df.rename(columns={self.id_col: self.ID_COL,
                  self.rel_col: self.LABEL_COL}, inplace=True)
        df.dropna(inplace=True)

        return df


class DummyDataHandler(DataHandler):
    def read_in_data(self, data_path: str) -> pd.DataFrame:
        # check if file is delimited by comma or semicolon
        with open(data_path, 'r') as f:
            first_line = f.readline()
            # check what character is before "labels"
            left, _ = first_line.split(self.LABEL_COL)
            delimiter = left[-1]
        df = pd.read_csv(data_path, delimiter=delimiter)
        # check if [ in labels --> make list
        try:
            if '[' in df[self.LABEL_COL].iloc[0]:
                df[self.LABEL_COL] = df[self.LABEL_COL].apply(
                    lambda x: x.strip('][').split(', ')
                )
        except TypeError:
            pass

        return df


def main():
    pass
    # TODO: Update this main function to test the DataHandler class
    # pseudopath = 'imaginary_file.jsonl'
    # # my_datahanlder = DataHandler(pseudopath)

    # # Test stratified split
    # my_datahandler = DataHandler()
    # train, test, val = my_datahandler.get_strat_split()
    # my_datahandler.save_split('./data/annotated_data/test_split')
    # my_second_datahandler = DataHandler()
    # my_second_datahandler.load_splits('./data/annotated_data/test_split')
    # train, test, val = my_second_datahandler.get_strat_split()

    # # Test stratified with val split
    # datahandler = DataHandler()
    # train, test, val = datahandler.get_strat_split(
    #     train_size=0.6, use_val=True)
    # datahandler.save_split('./data/annotated_data/test_split')
    # my_second_datahandler = DataHandler()
    # my_second_datahandler.load_splits('./data/annotated_data/test_split')
    # train, test, val = my_second_datahandler.get_strat_split(use_val=True)

    # # Test k-fold split
    # dataHandler = DataHandler()
    # dataHandler.get_strat_k_fold_split()
    # dataHandler.save_split('./data/annotated_data/test_split')
    # my_second_datahandler = DataHandler()
    # my_second_datahandler.load_splits('./data/annotated_data/test_split')

    # # Test k-fold split with val
    # dataHandler = DataHandler()
    # dataHandler.get_strat_k_fold_split(use_val=True)
    # dataHandler.save_split('./data/annotated_data/test_split')
    # my_second_datahandler = DataHandler()
    # my_second_datahandler.load_splits('./data/annotated_data/test_split')


if __name__ == '__main__':
    main()
