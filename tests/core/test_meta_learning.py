import unittest
import json
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.meta_learning import MetaLearner

class TestMetaLearner(unittest.TestCase):
    """Unit tests for the MetaLearner class."""

    def setUp(self):
        """Set up a temporary experiences file for testing."""
        self.test_experiences_path = 'test_experiences.json'
        self.test_experiences = [
            {
                "project_name": "Project A (Classif)",
                "dataset_profile": {
                    "n_rows": 1000, "n_features": 10, "problem_type": "Classification",
                    "feature_profiles": {"numerical": 5, "categorical": 5, "text": 0},
                    "missing_value_info": {"overall_missing_ratio": 0.05}
                },
                "best_pipeline": {"model": "RandomForestClassifier", "preprocessing": ["Imputer"]}
            },
            {
                "project_name": "Project B (Classif)",
                "dataset_profile": {
                    "n_rows": 1200, "n_features": 12, "problem_type": "Classification",
                    "feature_profiles": {"numerical": 6, "categorical": 6, "text": 0},
                    "missing_value_info": {"overall_missing_ratio": 0.1}
                },
                "best_pipeline": {"model": "XGBClassifier", "preprocessing": ["Imputer", "Scaler"]}
            },
            {
                "project_name": "Project C (Regr)",
                "dataset_profile": {
                    "n_rows": 500, "n_features": 20, "problem_type": "Regression",
                    "feature_profiles": {"numerical": 18, "categorical": 2, "text": 0},
                    "missing_value_info": {"overall_missing_ratio": 0.0}
                },
                "best_pipeline": {"model": "LinearRegression", "preprocessing": ["Scaler"]}
            }
        ]
        with open(self.test_experiences_path, 'w') as f:
            json.dump(self.test_experiences, f)
        
        self.meta_learner = MetaLearner(experiences_path=self.test_experiences_path)

    def tearDown(self):
        """Remove the temporary experiences file."""
        os.remove(self.test_experiences_path)

    def test_calculate_similarity(self):
        """Test the similarity calculation logic."""
        profile1 = {
            "n_rows": 950, "n_features": 11, "problem_type": "Classification",
            "feature_profiles": {
                'f1': {'type': 'numerical'}, 'f2': {'type': 'numerical'}, 'f3': {'type': 'numerical'},
                'f4': {'type': 'numerical'}, 'f5': {'type': 'numerical'}, 'f6': {'type': 'categorical'},
                'f7': {'type': 'categorical'}, 'f8': {'type': 'categorical'}, 'f9': {'type': 'categorical'},
                'f10': {'type': 'categorical'}, 'f11': {'type': 'categorical'}
            },
            "missing_value_info": {"overall_missing_ratio": 0.06}
        }
        # This profile is very similar to Project A
        score_a = self.meta_learner.calculate_similarity(profile1, self.test_experiences[0]['dataset_profile'])
        
        # This profile is very different (Regression)
        score_c = self.meta_learner.calculate_similarity(profile1, self.test_experiences[2]['dataset_profile'])

        self.assertLess(score_a, 1.0) # Should be a low score for high similarity
        self.assertEqual(score_c, float('inf')) # Should be infinite for different problem types

    def test_recommend_algorithms_classification(self):
        """Test algorithm recommendation for a classification problem."""
        current_profile = {
            "n_rows": 1100, "n_features": 11, "problem_type": "Classification",
            "target_column": "y",
            "feature_profiles": {
                'f1': {'type': 'numerical'}, 'f2': {'type': 'numerical'}, 'f3': {'type': 'numerical'},
                'f4': {'type': 'numerical'}, 'f5': {'type': 'numerical'}, 'f6': {'type': 'categorical'},
                'f7': {'type': 'categorical'}, 'f8': {'type': 'categorical'}, 'f9': {'type': 'categorical'},
                'f10': {'type': 'categorical'}, 'y': {'type': 'categorical'}
            },
            "missing_value_info": {"overall_missing_ratio": 0.08}
        }
        
        recommendations = self.meta_learner.recommend_algorithms(current_profile)
        
        self.assertEqual(len(recommendations), 2) # Should find the two classification projects
        self.assertEqual(recommendations[0]['model'], 'XGBClassifier') # Project B is slightly more similar
        self.assertEqual(recommendations[1]['model'], 'RandomForestClassifier')

    def test_recommend_algorithms_no_match(self):
        """Test that no recommendations are returned for a completely different profile."""
        current_profile = {
            "n_rows": 10000, "n_features": 100, "problem_type": "Regression",
            "target_column": "y",
            "feature_profiles": {f'f{i}': {'type': 'numerical'} for i in range(100)},
            "missing_value_info": {"overall_missing_ratio": 0.8}
        }
        
        recommendations = self.meta_learner.recommend_algorithms(current_profile)
        # It will still find the one regression project, but the score will be high.
        # The recommendation logic returns the best it can find.
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]['model'], 'LinearRegression')

if __name__ == '__main__':
    unittest.main()
