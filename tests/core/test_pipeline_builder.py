# tests/core/test_pipeline_builder.py

import unittest
import pandas as pd
from unittest.mock import MagicMock

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

from core.pipeline_builder import PipelineBuilder
from core.model_manager import ModelManager

class TestPipelineBuilder(unittest.TestCase):

    def setUp(self):
        """Set up a mock model manager and a sample dataframe."""
        self.mock_model_manager = ModelManager()
        # Let's mock get_estimator to be predictable
        self.mock_model_manager.get_estimator = MagicMock(
            return_value=DecisionTreeClassifier()
        )
        
        self.recommendations = [{'model': 'DecisionTreeClassifier', 'preprocessing': [], 'source_projects': []}]
        self.builder = PipelineBuilder(self.recommendations, self.mock_model_manager)

        self.sample_df = pd.DataFrame({
            'numerical_feature': [1, 2, 3, 4, 5],
            'categorical_feature': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
        self.dataset_profile = {
            'n_rows': 5,
            'n_features': 2,
            'problem_type': 'classification',
            'target_column': 'target'
        }

    def test_create_pipeline_simple(self):
        """Test creating a simple pipeline with a model name."""
        pipeline = self.builder.create_pipeline('DecisionTreeClassifier', [])
        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(len(pipeline.steps), 1)
        self.assertEqual(pipeline.steps[0][0], 'model')
        self.assertIsInstance(pipeline.steps[0][1], DecisionTreeClassifier)
        self.mock_model_manager.get_estimator.assert_called_with('DecisionTreeClassifier')

    def test_create_pipeline_from_profile(self):
        """Test creating a full preprocessing and model pipeline from a profile."""
        X_train = self.sample_df[['numerical_feature', 'categorical_feature']]
        
        pipeline = self.builder.create_pipeline_from_profile(
            'DecisionTreeClassifier', self.dataset_profile, X_train
        )

        # 1. Check overall structure
        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(len(pipeline.steps), 2)
        self.assertEqual(pipeline.steps[0][0], 'preprocessor')
        self.assertEqual(pipeline.steps[1][0], 'model')

        # 2. Check the preprocessor (ColumnTransformer)
        preprocessor = pipeline.named_steps['preprocessor']
        self.assertIsInstance(preprocessor, ColumnTransformer)
        self.assertEqual(len(preprocessor.transformers), 2) # one for numerical, one for categorical

        # 3. Check the model
        model = pipeline.named_steps['model']
        self.assertIsInstance(model, DecisionTreeClassifier)

        # 4. Check that the correct columns are being transformed
        num_transformer_cols = preprocessor.transformers[0][2]
        cat_transformer_cols = preprocessor.transformers[1][2]
        self.assertEqual(num_transformer_cols, ['numerical_feature'])
        self.assertEqual(cat_transformer_cols, ['categorical_feature'])

    def test_create_pipeline_with_no_categorical_features(self):
        """Test pipeline creation when there are no categorical features."""
        X_train = self.sample_df[['numerical_feature']]
        
        pipeline = self.builder.create_pipeline_from_profile(
            'DecisionTreeClassifier', self.dataset_profile, X_train
        )
        
        preprocessor = pipeline.named_steps['preprocessor']
        self.assertEqual(len(preprocessor.transformers), 1)
        self.assertEqual(preprocessor.transformers[0][0], 'numerical')

    def test_create_pipeline_with_no_numerical_features(self):
        """Test pipeline creation when there are no numerical features."""
        X_train = self.sample_df[['categorical_feature']]
        
        pipeline = self.builder.create_pipeline_from_profile(
            'DecisionTreeClassifier', self.dataset_profile, X_train
        )
        
        preprocessor = pipeline.named_steps['preprocessor']
        self.assertEqual(len(preprocessor.transformers), 1)
        self.assertEqual(preprocessor.transformers[0][0], 'categorical')

if __name__ == '__main__':
    unittest.main()
