import unittest
import pandas as pd
from data.iaa import *

class TestCalculateOverallCohenKappa(unittest.TestCase):

    def test_calculate_overall_cohen_kappa_with_ci(self):
        # Create a sample DataFrame
        data = {
            'annotations_array_numeric_annotator1': [[1, 2, 3], [2, 3, 1]],
            'annotations_array_numeric_annotator2': [[1, 2, 3], [1, 2, 3]]
        }
        df = pd.DataFrame(data)

        annotators = ['annotator1', 'annotator2']

        # Call the function
        calculate_overall_cohen_kappa_with_ci(df, annotators)

        # Add your assertions here if applicable


if __name__ == '__main__':
    unittest.main()


