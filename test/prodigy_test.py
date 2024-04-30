import unittest
import pandas as pd
import os
from data.prodigy_data_reader import *


class TestCalculateOverallCohenKappa(unittest.TestCase):

    def setUp(self):
        self.relative_path = os.path.dirname(__file__)
        prodigy_export = 'test_data/test_export.jsonl'
        self.prodigy_reader = ProdigyDataReader(
            os.path.join(self.relative_path, prodigy_export))


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
    
    def test_get_task_df(self):
        # check if the following raises a ValueErrror
        self.assertRaises(ValueError, self.prodigy_reader.get_task_df, '42')
        
        test_task = 'Study Type'
        task_df = self.prodigy_reader.get_task_df(test_task)
        # check if the returned object is a DataFrame
        self.assertIsInstance(task_df, pd.DataFrame)
        
        expected_test_task_columns = self.prodigy_reader.get_classification_tasks()[test_task]
        expected_test_task_columns += FIXED_COLUMNS
        # check if the excpected columns are in the DataFrame
        for column in task_df.columns:
            self.assertIn(column, expected_test_task_columns)
        
        # cherry pick a line from the DataFrame, with id==7601
        expected_test_column1 = {
            'Randomized-controlled trial (RCT)' : 0,
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
        task2_df = self.prodigy_reader.get_task_df(test_task2)
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
        task_df = self.prodigy_reader.get_task_df(test_task)
        test_plot = os.path.join(self.relative_path, 'test_data/test_plot.png')
        self.prodigy_reader.visualize_distribution(task_df, test_task, test_plot)
        # check if the file was created
        self.assertTrue(os.path.exists(test_plot))
    
if __name__ == '__main__':
    unittest.main()
