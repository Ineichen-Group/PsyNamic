import os
from abc import abstractmethod
from os import path
from typing import Iterable, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset

# Fix the random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


class DataSplit(Dataset):
    ID_COL = 'id'
    TEXT_COL = 'text'
    LABEL_COL = 'labels'

    def __init__(self, split: pd.DataFrame) -> None:
        self.df = split

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        text = self.df.iloc[idx][self.TEXT_COL]
        labels = self.df.iloc[idx][self.LABEL_COL]
        return {self.TEXT_COL: text, self.LABEL_COL: labels}


class DataHandler():
    ID_COL = 'id'
    TEXT_COL = 'text'
    LABEL_COL = 'labels'

    def __init__(self, data_path: str = None, nr_classes: int = None, is_multilabel: bool = None, int_to_label: dict = None) -> None:
        if data_path:
            self.df = self.read_in_data(data_path)
        else:
            self.df = None
        # Create a dummy df for testing, with three columns: id, text, labels and 3 dummy rows
        # Generate dummy data
        data = {
            self.ID_COL: list(range(1, 21)),  # IDs from 1 to 20
            self.TEXT_COL: [
                'Text number {}'.format(i) for i in range(1, 21)
            ],  # Texts with placeholder numbers
            # Randomly select labels 0, 1, or 2
            self.LABEL_COL: np.random.choice([0, 1, 2], size=20)
        }

        self.df = pd.DataFrame(data)
        self.is_multilabel = is_multilabel if is_multilabel else self._check_if_multilabel()
        self.nr_classes = nr_classes if nr_classes else self._determine_nr_classes()
        self.int_to_label = int_to_label

        self.train = None
        self.test = None
        self.val = None
        self.fold = []

    def __len__(self) -> int:
        return len(self.df)

    def _determine_nr_classes(self) -> int:
        if self.is_multilabel:
            raise NotImplementedError(
                'Multilabel classification not implemented yet.')
        else:
            return len(self.df[self.LABEL_COL].unique())

    def _check_if_multilabel(self) -> bool:
        return any(isinstance(label, list) for label in self.df[self.LABEL_COL])

    def _check_presence_of_labels(self, datasplit: pd.DataFrame) -> bool:
        if self.is_multilabel:
            raise NotImplementedError(
                'Multilabel classification not implemented yet.')
        else:
            return self.nr_classes == len(datasplit[self.LABEL_COL].unique())

    def get_strat_split(self, train_size: float = 0.8, use_val: bool = False, resplit: bool = False) -> tuple[DataSplit, DataSplit, Union[DataSplit, None]]:
        if self.is_multilabel:
            raise NotImplementedError(
                'Multilabel stratified split not implemented yet.')
        else:
            if use_val:
                if self.train is not None and not resplit:
                    return DataSplit(self.train), DataSplit(self.test), DataSplit(self.val)

                # First split: train (60%) and remaining (40%)
                split = StratifiedShuffleSplit(
                    n_splits=1, train_size=train_size, test_size=1-train_size, random_state=SEED)
                train_idx, remaining_idx = next(
                    split.split(self.df, self.df[self.LABEL_COL]))
                train_df = self.df.iloc[train_idx]
                remaining_df = self.df.iloc[remaining_idx]

                # Second split: remaining (40%) into val (20%) and test (20%)
                val_test_split = StratifiedShuffleSplit(
                    n_splits=1, train_size=0.5, test_size=0.5, random_state=SEED)
                val_idx, test_idx = next(val_test_split.split(
                    remaining_df, remaining_df[self.LABEL_COL]))
                val_df = remaining_df.iloc[val_idx]
                test_df = remaining_df.iloc[test_idx]
                if not (self._check_presence_of_labels(train_df) and self._check_presence_of_labels(val_df) and self._check_presence_of_labels(test_df)):
                    raise ValueError(
                        'Not all labels are present in the train, val and test set.')
                self.train = train_df
                self.val = val_df
                self.test = test_df
                return DataSplit(train_df), DataSplit(test_df), DataSplit(val_df)
            else:
                if self.train is not None and not resplit:
                    return DataSplit(self.train), DataSplit(self.test), None

                # Direct split: train (80%) and test (20%)
                split = StratifiedShuffleSplit(
                    n_splits=1, train_size=train_size, random_state=SEED)
                train_idx, test_idx = next(split.split(
                    self.df, self.df[self.LABEL_COL]))
                train_df = self.df.iloc[train_idx]
                test_df = self.df.iloc[test_idx]
                self.train = train_df
                self.test = test_df

                if not (self._check_presence_of_labels(train_df) and self._check_presence_of_labels(test_df)):
                    raise ValueError(
                        'Not all labels are present in the train and test set.')

                return DataSplit(train_df), DataSplit(test_df), None

    def get_strat_k_fold_split(self, train_size: float = 0.8, n_splits: int = 5, use_val: bool = False) -> Iterable[tuple[DataSplit, DataSplit, Union[DataSplit, None]]]:

        if self.is_multilabel:
            raise NotImplementedError(
                'Multilabel stratified split not implemented yet.')
        else:
            # Check if the folds have already been created, if so return them
            if self.folds:
                return self.folds

            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=SEED)
            folds = []
            for train_idx, test_idx in skf.split(self.df, self.df[self.LABEL_COL]):
                train_df = self.df.iloc[train_idx]
                test_df = self.df.iloc[test_idx]

                if use_val:
                    # Split train_df into actual train and validation
                    train_val_split = StratifiedShuffleSplit(
                        n_splits=1, train_size=train_size, test_size=1-train_size, random_state=SEED)
                    actual_train_idx, val_idx = next(
                        train_val_split.split(train_df, train_df[self.LABEL_COL]))
                    actual_train_df = train_df.iloc[actual_train_idx]
                    val_df = train_df.iloc[val_idx]
                    folds.append(DataSplit(actual_train_df),
                                 DataSplit(test_df)), DataSplit(val_df)
                else:
                    folds.append(DataSplit(train_df), DataSplit(test_df), None)

                self.folds = folds
                return folds

    def save_split(self, save_path: str) -> None:
        if not path.exists(save_path):
            raise FileNotFoundError(f'Path {save_path} does not exist.')
        if self.fold:
            for i, fold in enumerate(self.fold):
                fold[0].to_csv(f'{save_path}/train_fold_{i}.csv', index=False)
                fold[1].to_csv(f'{save_path}/test_fold_{i}.csv', index=False)
                try:
                    fold[2].to_csv(
                        f'{save_path}/val_fold_{i}.csv', index=False)
                except KeyError:
                    pass
        else:
            self.train.to_csv(path.join(save_path, 'train.csv'), index=False)
            self.test.to_csv(path.join(save_path, 'test.csv'), index=False)
            try:
                self.val.to_csv(path.join(save_path, 'val.csv'), index=False)
            except AttributeError:
                pass

    def load_splits(self, load_path: str) -> None:
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
                    self.fold.append((train_data, test_data, val_data))
            else:
                raise FileNotFoundError(
                    f'No train/test/val files found in {load_path}.')

    @abstractmethod
    def preprocess(self) -> None:
        pass

    @abstractmethod
    def read_in_data(self, data_path: str) -> pd.DataFrame:
        pass


class PsychNamicDataHandler(DataHandler):

    def __init__(self, data: pd.DataFrame, label: str) -> None:
        pass

    def preprocess(self) -> None:
        pass

    def read_in_data(self, data_path: str) -> pd.DataFrame:
        pass


def main():
    pseudopath = 'imaginary_file.jsonl'
    # my_datahanlder = DataHandler(pseudopath)

    # train, test, val = my_datahanlder.stratified_split()
    # my_datahanlder.save_split('./data/annotated_data/test_split')
    # my_second_datahandler = DataHandler()
    # my_second_datahandler.load_splits('./data/annotated_data/test_split')
    # train, test, val = my_second_datahandler.get_strat_split()
   
    
    datahandler = DataHandler()
    train, test, val = datahandler.get_strat_split(train_size=0.6, use_val=True)
    datahandler.save_split('./data/annotated_data/test_split')
    my_second_datahandler = DataHandler()
    my_second_datahandler.load_splits('./data/annotated_data/test_split')
    train, test, val = my_second_datahandler.get_strat_split(use_val=True)
    print('blibal')
    


if __name__ == '__main__':
    main()
