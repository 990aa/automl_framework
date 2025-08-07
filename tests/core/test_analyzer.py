import unittest
import pandas as pd
import numpy as np
from core.data_analysis import DataAnalyzer

class TestDataAnalyzer(unittest.TestCase):
    """Unit tests for the DataAnalyzer class."""

    def setUp(self):
        """Set up a new DataAnalyzer for each test."""
        self.analyzer = DataAnalyzer()

    def test_classification_problem(self):
        """Test that a dataset is correctly identified as a classification problem."""
        data = {
            'feature1': [1.2, 2.3, 3.4, 4.5, 5.6],
            'feature2': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(data)
        profile = self.analyzer.analyze_dataset(df)
        self.assertEqual(profile['problem_type'], 'Classification')
        self.assertEqual(profile['target_column'], 'target')
        self.assertEqual(profile['feature_profiles']['feature1']['type'], 'numerical')
        self.assertEqual(profile['feature_profiles']['feature2']['type'], 'categorical')

    def test_regression_problem(self):
        """Test that a dataset is correctly identified as a regression problem."""
        data = {
            'feature1': [10, 20, 30, 40, 50],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'y': [15.5, 25.5, 35.5, 45.5, 55.5]
        }
        df = pd.DataFrame(data)
        profile = self.analyzer.analyze_dataset(df)
        self.assertEqual(profile['problem_type'], 'Regression')
        self.assertEqual(profile['target_column'], 'y')

    def test_missing_values(self):
        """Test the missing value analysis."""
        data = {
            'col_a': [1, 2, np.nan, 4],
            'col_b': [np.nan, 'B', 'C', 'D']
        }
        df = pd.DataFrame(data)
        profile = self.analyzer.analyze_dataset(df)
        
        # Overall missing ratio: 2 missing out of 8 cells = 0.25
        self.assertAlmostEqual(profile['missing_value_info']['overall_missing_ratio'], 0.25)
        
        # Per-column missing ratio
        self.assertAlmostEqual(profile['missing_value_info']['missing_ratio_per_column']['col_a'], 0.25)
        self.assertAlmostEqual(profile['feature_profiles']['col_a']['missing_ratio'], 0.25)
        self.assertAlmostEqual(profile['missing_value_info']['missing_ratio_per_column']['col_b'], 0.25)

    def test_near_zero_variance(self):
        """Test the detection of near-zero variance features."""
        data = {
            'feature_stable': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 10,
            'feature_diverse': range(100)
        }
        df = pd.DataFrame(data)
        profile = self.analyzer.analyze_dataset(df)
        self.assertIn('feature_stable', profile['data_quality']['near_zero_variance_features'])
        self.assertNotIn('feature_diverse', profile['data_quality']['near_zero_variance_features'])

    def test_text_feature_detection(self):
        """Test the detection of text features."""
        data = {
            'short_string': ['cat', 'dog', 'rat', 'cat'],
            'long_text': [
                "this is a very long sentence that should be classified as text",
                "another long piece of text to ensure the heuristic works well",
                "short",
                "even more extensive text data to satisfy the length condition"
            ]
        }
        df = pd.DataFrame(data)
        profile = self.analyzer.analyze_dataset(df)
        self.assertEqual(profile['feature_profiles']['short_string']['type'], 'categorical')
        self.assertEqual(profile['feature_profiles']['long_text']['type'], 'text')

    def test_unsupervised_problem(self):
        """Test detection of unsupervised problem when no clear target exists."""
        data = {
            'sensor_1': [0.1, 0.2, 0.1, 0.3],
            'sensor_2': [0.5, 0.4, 0.5, 0.6]
        }
        df = pd.DataFrame(data)
        # With the aggressive fallback removed, this should now be correctly identified as Unsupervised.
        profile = self.analyzer.analyze_dataset(df)
        self.assertEqual(profile['problem_type'], 'Unsupervised')
        
        # This second test is now redundant, but we'll keep it to be explicit.
        analyzer_no_fallback = DataAnalyzer(target_candidates=['non_existent_column'])
        profile_no_fallback = analyzer_no_fallback.analyze_dataset(df)
        self.assertEqual(profile_no_fallback['problem_type'], 'Unsupervised')


if __name__ == '__main__':
    unittest.main()
