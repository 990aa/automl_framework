import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .model_manager import ModelManager

class PipelineBuilder:
    def __init__(self, recommendations, model_manager):
        self.recommendations = recommendations
        self.model_manager = model_manager

    def create_pipeline(self, algorithm_name, preprocessing_steps):
        """
        Creates a scikit-learn pipeline with the given algorithm and preprocessing steps.
        """
        algorithm = self.model_manager.get_model(algorithm_name)
        if algorithm is None:
            raise ValueError(f"Algorithm '{algorithm_name}' not found.")

        steps = preprocessing_steps + [('classifier', algorithm)]
        return Pipeline(steps)

def handle_missing(df, strategy='mean'):
    return SimpleImputer(strategy=strategy)

def scale_numerical(df):
    return StandardScaler()

def encode_categorical(df, handle_unknown='ignore'):
    return OneHotEncoder(handle_unknown=handle_unknown)

def text_features_processing(df, steps=[]):
    return ColumnTransformer(steps)