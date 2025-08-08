from dask_ml.preprocessing import StandardScaler, Categorizer, DummyEncoder
from dask_ml.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd

class PipelineBuilder:
    """
    Builds machine learning pipelines based on dataset profiles and recommendations.
    """
    def __init__(self, recommendations, model_manager):
        """
        Initializes the PipelineBuilder.

        Args:
            recommendations (list): A list of recommended algorithms.
            model_manager (ModelManager): The model manager to use for getting estimators.
        """
        self.recommendations = recommendations
        self.model_manager = model_manager

    def create_pipeline_from_profile(self, model_name, dataset_profile, X_train):
        """
        Builds a Dask-compatible pipeline using a dataset profile.

        Args:
            model_name (str): The name of the model to use in the pipeline.
            dataset_profile (dict): A dictionary containing the dataset profile.
            X_train (dask.dataframe.DataFrame): The training features.

        Returns:
            sklearn.pipeline.Pipeline: The constructed pipeline.
        
        Raises:
            ValueError: If the requested estimator is not found.
        """
        numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        preprocessor_steps = []
        if numerical_cols:
            # Pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            preprocessor_steps.append(('numerical', num_pipeline, numerical_cols))

        if categorical_cols:
            # Pipeline for categorical features
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('categorizer', Categorizer()), # Converts strings to integer categories
                ('encoder', DummyEncoder())    # One-hot encodes the categories
            ])
            preprocessor_steps.append(('categorical', cat_pipeline, categorical_cols))

        preprocessor = ColumnTransformer(transformers=preprocessor_steps)
        
        estimator = self.model_manager.get_estimator(model_name)
        if not estimator:
            raise ValueError(f"Estimator '{model_name}' not found.")

        # Final pipeline with preprocessor and Dask-ML model
        return Pipeline(steps=[('preprocessor', preprocessor), ('model', estimator)])