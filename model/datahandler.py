import os
from abc import abstractmethod
from os import path
from typing import Iterable, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

# Fix the random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


class DataSplit(Dataset):
    "PyTorch Dataset class for a given data split."
    ID_COL = 'id'
    TEXT_COL = 'text'
    LABEL_COL = 'labels'

    def __init__(self, split: pd.DataFrame, multilabel: bool = False) -> None:
        self.df = split
        self.is_multilabel = multilabel

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        text = self.df.iloc[idx][self.TEXT_COL]
        labels = self.df.iloc[idx][self.LABEL_COL]

        # in case of multilabel classification, convert labels to tensor
        if isinstance(labels, list):
            labels = torch.tensor(labels)

        return {self.TEXT_COL: text, self.LABEL_COL: labels}

    def __eq__(self, other) -> bool:
        if not isinstance(other, DataSplit):
            return False
        return self.df.equals(other.df)

    def __repr__(self) -> str:
        return f"DataSplit(num_samples={len(self.df)}, labels={self.labels})"

    def to_csv(self, save_path: str) -> None:
        self.df.to_csv(save_path, index=False)

    @property
    def labels(self) -> list[int]:
        if self.is_multilabel:
            label_tuples = self.df['labels'].apply(tuple)
            unique_labels = set(label_tuples)
            return list(unique_labels)
        else:
            label_list = self.df[self.LABEL_COL].unique().tolist()
            label_list.sort()
            return label_list


class DataHandler():
    """ Abstract DataHandler class to handle data loading, preprocessing and splitting.
        Idea: Inherit from this class and implement the abstract methods to create a DataHandler for a specific dataset.
    """
    ID_COL = 'id'
    TEXT_COL = 'text'
    LABEL_COL = 'labels'

    def __init__(self, data_path: str = None, nr_classes: int = None, is_multilabel: bool = None) -> None:
        if data_path:
            self.df = self.read_in_data(data_path)
        else:
            self.df = None

        self.is_multilabel = is_multilabel if is_multilabel else self._check_if_multilabel()
        self.nr_classes = nr_classes if nr_classes else self._determine_nr_classes()

        self.train = None
        self.test = None
        self.val = None
        self.folds = []
        self.train_size = None
        self.use_val = None

    def __len__(self) -> int:
        return len(self.df)

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
            if use_val:
                return DataSplit(self.train), DataSplit(self.test), DataSplit(self.val)
            else:
                return DataSplit(self.train), DataSplit(self.test), None
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
                return DataSplit(train_df), DataSplit(test_df), DataSplit(val_df)

            else:
                msss = MultilabelStratifiedShuffleSplit(
                    n_splits=1, test_size=1-train_size, random_state=SEED)
                train_index, test_index = next(msss.split(X, y))
                train_df = self.df.iloc[train_index]
                test_df = self.df.iloc[test_index]
                # Save splits to avoid recomputation
                self.train = train_df
                self.test = test_df

                return DataSplit(train_df), DataSplit(test_df), None
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
                if not (self._check_presence_of_labels(train_df) and self._check_presence_of_labels(val_df) and self._check_presence_of_labels(test_df)):
                    raise ValueError(
                        'Not all labels are present in the train, val and test set; data set might be too small or there is an error in the code.')
                self.train = train_df
                self.val = val_df
                self.test = test_df
                return DataSplit(train_df), DataSplit(test_df), DataSplit(val_df)
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

                if not (self._check_presence_of_labels(train_df) and self._check_presence_of_labels(test_df)):
                    raise ValueError(
                        'Not all labels are present in the train and test set; data set might be too small or there is an error in the code.')

                return DataSplit(train_df), DataSplit(test_df), None

    def get_strat_k_fold_split(self, train_size: float = 0.8, n_splits: int = 5, seed: int = SEED) -> tuple[Iterable[tuple[DataSplit, DataSplit]], DataSplit]:
        """ Get stratified k-fold split of the data; i.e. n splits into train and validation set, with a test set.

        Args:
            train_size (float, optional): Size of the training set. Defaults to 0.8.
            n_splits (int, optional): Number of splits. Defaults to 5.
            seed (int, optional): Random seed. Defaults to SEED.

        Returns:
            tuple[Iterable[tuple[DataSplit, DataSplit]], DataSplit]: Iterable of k-folds and test split.
        """
        train_split, test_split, _ = self.get_strat_split(
            train_size=train_size)

        if self.is_multilabel:
            mskf = MultilabelStratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=SEED)
            folds = []
            X = train_split.df[self.TEXT_COL].values
            y = np.array(train_split.df[self.LABEL_COL].tolist())

            for train_idx, val_idx in mskf.split(X, y):
                train_df = train_split.df.iloc[train_idx]
                val_df = train_split.df.iloc[val_idx]
                folds.append(
                    (DataSplit(train_df), DataSplit(val_df)))

            self.folds = folds
            return folds, test_split

        else:
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=SEED)
            folds = []
            for train_idx, val_idx in skf.split(train_split.df, train_split.df[self.LABEL_COL]):
                train_df = train_split.df.iloc[train_idx]
                val_df = train_split.df.iloc[val_idx]

                folds.append(
                    (DataSplit(train_df), DataSplit(val_df)))

            self.folds = folds
            return folds, test_split

    def save_split(self, save_path: str) -> None:
        """ Save the train, test and validation splits to a given path as csv files.
        """
        if not path.exists(save_path):
            raise FileNotFoundError(f'Path {save_path} does not exist.')
        if self.folds:
            for i, fold in enumerate(self.folds):
                fold[0].to_csv(f'{save_path}/train_fold_{i}.csv', index=False)
                fold[1].to_csv(f'{save_path}/test_fold_{i}.csv', index=False)
                try:
                    fold[2].to_csv(
                        f'{save_path}/val_fold_{i}.csv', index=False)
                except AttributeError:
                    pass
        else:
            self.train.to_csv(path.join(save_path, 'train.csv'), index=False)
            self.test.to_csv(path.join(save_path, 'test.csv'), index=False)
            try:
                self.val.to_csv(path.join(save_path, 'val.csv'), index=False)
            except AttributeError:
                pass

    def load_splits(self, load_path: str) -> None:
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
            else:
                self.df = pd.concat([self.train, self.test])
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


class PsyNamic1vsAll(DataHandler):

    def __init__(self, data_path: str, relevant_class: str,  nr_classes: int = None, is_multilabel: bool = None, int_to_label: dict = None) -> None:
        super().__init__(data_path, nr_classes, is_multilabel, int_to_label)
        self.relevant_class = relevant_class

    def read_in_data(self, data_path: str) -> pd.DataFrame:
        pass


class PsychNamicBIOHandler(DataHandler):
    pass
    
    def read_in_data(self, data_path: str) -> pd.DataFrame:
        pass


class PsychNamicRelevant(DataHandler):
    pass

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
    pseudopath = 'imaginary_file.jsonl'
    # my_datahanlder = DataHandler(pseudopath)

    # Test stratified split
    my_datahandler = DataHandler()
    train, test, val = my_datahandler.get_strat_split()
    my_datahandler.save_split('./data/annotated_data/test_split')
    my_second_datahandler = DataHandler()
    my_second_datahandler.load_splits('./data/annotated_data/test_split')
    train, test, val = my_second_datahandler.get_strat_split()

    # Test stratified with val split
    datahandler = DataHandler()
    train, test, val = datahandler.get_strat_split(
        train_size=0.6, use_val=True)
    datahandler.save_split('./data/annotated_data/test_split')
    my_second_datahandler = DataHandler()
    my_second_datahandler.load_splits('./data/annotated_data/test_split')
    train, test, val = my_second_datahandler.get_strat_split(use_val=True)

    # Test k-fold split
    dataHandler = DataHandler()
    dataHandler.get_strat_k_fold_split()
    dataHandler.save_split('./data/annotated_data/test_split')
    my_second_datahandler = DataHandler()
    my_second_datahandler.load_splits('./data/annotated_data/test_split')

    # Test k-fold split with val
    dataHandler = DataHandler()
    dataHandler.get_strat_k_fold_split(use_val=True)
    dataHandler.save_split('./data/annotated_data/test_split')
    my_second_datahandler = DataHandler()
    my_second_datahandler.load_splits('./data/annotated_data/test_split')


if __name__ == '__main__':
    main()
