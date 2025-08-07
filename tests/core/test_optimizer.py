# tests/core/test_optimizer.py

import unittest
import dask.dataframe as dd
import pandas as pd
from unittest.mock import patch, MagicMock

from sklearn.pipeline import Pipeline
from dask_ml.linear_model import LogisticRegression
from dask_ml.model_selection import GridSearchCV

from core.optimizer import Optimizer

class TestDaskOptimizer(unittest.TestCase):

    def setUp(self):
        """Set up a Dask-compatible environment for testing."""
        self.optimizer = Optimizer()
        
        # Create sample Dask DataFrames
        pdf = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [0, 1, 0, 1, 0, 1, 0, 1],
            'target': [0, 1, 0, 1, 0, 1, 0, 1]
        })
        ddf = dd.from_pandas(pdf, npartitions=2)
        
        self.X = ddf[['feature1', 'feature2']]
        self.y = ddf['target']
        
        # A simple pipeline template
        self.pipeline_template = Pipeline([
            ('model', LogisticRegression())
        ])
        
        # A simple search grid
        self.search_grid = self.optimizer.get_default_grid('LogisticRegression')

    def test_get_default_grid_prefixes_correctly(self):
        """Test that the grid parameters are correctly prefixed with 'model__'."""
        grid = self.optimizer.get_default_grid('LogisticRegression')
        self.assertIn('model__C', grid)
        self.assertNotIn('C', grid)

    @patch('core.optimizer.GridSearchCV')
    def test_optimize_hyperparameters_with_grid(self, MockGridSearchCV):
        """Test the main HPO logic by mocking Dask-ML's GridSearchCV."""
        # Configure the mock GridSearchCV instance
        mock_instance = MockGridSearchCV.return_value
        mock_instance.best_estimator_ = self.pipeline_template  # Return the template itself
        mock_instance.best_score_ = 0.95
        mock_instance.best_params_ = {'model__C': 1.0}
        
        # Mock the predict method on the returned pipeline
        # The pipeline needs a `predict` method that returns a Dask array
        mock_instance.best_estimator_.predict = MagicMock(return_value=self.X['feature2'])

        # Run the optimizer
        results = self.optimizer.optimize_hyperparameters(
            self.pipeline_template, self.X, self.y, self.X, self.y,
            self.search_grid, 'classification'
        )

        # 1. Assert GridSearchCV was called correctly
        MockGridSearchCV.assert_called_once_with(
            self.pipeline_template, self.search_grid, cv=3, scoring="accuracy"
        )
        
        # 2. Assert fit was called on the GridSearchCV instance
        mock_instance.fit.assert_called_once()

        # 3. Check the structure and content of the returned dictionary
        self.assertIsInstance(results, dict)
        self.assertIn('best_pipeline', results)
        self.assertEqual(results['best_cv_score'], 0.95)
        self.assertEqual(results['best_params']['model__C'], 1.0)
        self.assertGreater(results['test_accuracy'], 0)
        self.assertGreaterEqual(results['hpo_time'], 0)

    def test_optimize_hyperparameters_no_grid(self):
        """Test the fallback behavior when no search_grid is provided."""
        # Mock the pipeline's fit and predict methods
        self.pipeline_template.fit = MagicMock()
        self.pipeline_template.predict = MagicMock(return_value=self.X['feature2'])

        # Run the optimizer with no grid
        results = self.optimizer.optimize_hyperparameters(
            self.pipeline_template, self.X, self.y, self.X, self.y,
            None, 'classification'
        )

        # 1. Assert the base pipeline's fit method was called
        self.pipeline_template.fit.assert_called_with(self.X, self.y)

        # 2. Check the results dictionary
        self.assertEqual(results['best_pipeline'], self.pipeline_template)
        self.assertIsNone(results['best_params'])
        self.assertEqual(results['hpo_time'], 0)
        self.assertGreater(results['test_accuracy'], 0)

if __name__ == '__main__':
    unittest.main()