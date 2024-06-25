import os
import unittest
import pandas as pd
from data.prodigy_data_reader import *

# Run single test:
# python -m unittest test.prodigy_test.TestParseProdigyData.test_get_classification_tasks


class TestParseProdigyData(unittest.TestCase):

    def setUp(self):
        self.relative_path = os.path.dirname(__file__)
        prodigy_export = 'test_data/test_export.jsonl'
        self.prodigy_reader = ProdigyDataReader(
            os.path.join(self.relative_path, prodigy_export))
        prodigy_export_thematic = 'test_data/test_export_with_thematic_split.jsonl'
        self.prodigy_reader_thematic = ProdigyDataReader(
            os.path.join(self.relative_path, prodigy_export_thematic)
        )

    def tearDown(self):
        # if there's a file ending in reorered.jsonl in test-data, delete it
        for file in os.listdir(os.path.join(self.relative_path, 'test_data')):
            if file.endswith('reordered.jsonl'):
                os.remove(os.path.join(self.relative_path, 'test_data', file))

    def test_get_classification_tasks(self):
        classification_tasks = self.prodigy_reader.get_classification_tasks()

        # check if classification is a dict
        self.assertIsInstance(classification_tasks, dict)

        # check if both 8 keys
        self.assertEqual(len(classification_tasks), 17)

    def test_get_prodigy_label_map(self):
        prodigy_label_map = self.prodigy_reader.get_prodigy_label_map()

        prodigy_label_keys = [x for x in range(0, 122)]

        self.assertEqual(list(prodigy_label_map.keys()), prodigy_label_keys)

    def test_read_in_data(self):
        pass

    def test_get_onehot_task_df(self):
        # check if the following raises a ValueErrror
        self.assertRaises(
            ValueError, self.prodigy_reader.get_onehot_task_df, '42')

        test_task = 'Study Type'
        task_df = self.prodigy_reader.get_onehot_task_df(test_task)
        # check if the returned object is a DataFrame
        self.assertIsInstance(task_df, pd.DataFrame)

        expected_test_task_columns = self.prodigy_reader.get_classification_tasks()[
            test_task]
        expected_test_task_columns += FIXED_COLUMNS
        # check if the excpected columns are in the DataFrame
        for column in task_df.columns:
            self.assertIn(column, expected_test_task_columns)

        # cherry pick a line from the DataFrame, with id==7601
        expected_test_column1 = {
            'Randomized-controlled trial (RCT)': 0,
            'Cohort study': 0,
            'Real-world study': 0,
            'Study protocol': 0,
            'Systematic review/meta-analysis': 0,
            'Qualitative Study': 0,
            'Case series': 0,
            'Case report': 0,
            'Other': 1,
        }

        test_line = task_df[task_df['id'] == 7601]
        # check if line corresponds to the expected values
        for key, value in expected_test_column1.items():
            self.assertEqual(test_line[key].values[0], value)

        test_task2 = 'Condition'
        task2_df = self.prodigy_reader.get_onehot_task_df(test_task2)
        test_line2 = task2_df[task2_df['id'] == 7601]

        expected_test_column2 = {
            'Psychiatric condition': 0,
            'Depression': 0,
            'Anxiety': 0,
            'Post-traumatic stress disorder (PTSD)': 0,
            'Alcoholism': 0,
            'Other addictions (e.g. smoking)': 0,
            'Anorexia': 0,
            'Alzheimer\u2019s disease': 0,
            'Non-Alzheimer dementia': 0,
            'Substance abuse': 0,
            '(Chronic) Pain': 0,
            'Palliative Setting': 0,
            'Recreational Drug Use': 1,
            'Healthy Participants': 1,
        }

        for key, value in expected_test_column2.items():
            self.assertEqual(test_line2[key].values[0], value)

    def test_visualize_distribution(self):
        test_task = 'Study Type'
        test_plot = os.path.join(self.relative_path, 'test_data/test_plot.png')
        self.prodigy_reader.visualize_distribution(save_path=test_plot)
        # check if the file was created
        self.assertTrue(os.path.exists(test_plot))

    def test_get_label_task_df(self):
        # check if the following raises a ValueErrror
        self.assertRaises(
            ValueError, self.prodigy_reader.get_label_task_df, '42')

        test_task = 'Study Purpose'
        label_mapping, task_df = self.prodigy_reader.get_label_task_df(
            test_task)
        # check if the returned object is a DataFrame
        self.assertIsInstance(task_df, pd.DataFrame)
        self.assertIsInstance(label_mapping, dict)

        # check validity of label_mapping
        self.assertEqual(len(label_mapping), len(
            self.prodigy_reader.get_classification_tasks()[test_task]))
        self.assertTrue(all(isinstance(key, int)
                        for key in label_mapping.keys()))

        # check some values in task_df
        test_line = task_df[task_df['id'] == 8868]
        expected = [0, 2, 3]
        self.assertListEqual(test_line[test_task].values.tolist()[0], expected)

        test_line = task_df[task_df['id'] == 8909]
        excepted = [3]
        self.assertListEqual(test_line[test_task].values.tolist()[0], excepted)

    def test_thematic_split(self):

        self.assertTrue(self.prodigy_reader_thematic.has_thematic_split())
        self.assertFalse(self.prodigy_reader.has_thematic_split())

        # check if the following raises a ValueErrror
        self.assertRaises(ValueError,  ProdigyDataReader, os.path.join(
                self.relative_path, 'test_data/test_export_with_thematic_split_unordered_missing.jsonl')
        )
        prodigy_reader_unorderd = ProdigyDataReader(
            os.path.join(
                self.relative_path, 'test_data/test_export_with_thematic_split_unordered.jsonl')
        )
        self.assertTrue(prodigy_reader_unorderd.has_thematic_split())
        self.assertTrue(prodigy_reader_unorderd.jsonl_path.endswith('reordered.jsonl'))
        # check if the file was created
        self.assertTrue(os.path.exists(prodigy_reader_unorderd.jsonl_path))


if __name__ == '__main__':
    unittest.main()
