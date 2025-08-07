# Automated Pipeline Creation & Basic Training
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

class PipelineBuilder:
    """
    Builds ML pipelines with recommended algorithms and preprocessing steps.
    """
    def __init__(self):
        pass

    def handle_missing(self, df, strategy='mean'):
        """
        Returns a SimpleImputer for numerical columns.
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return None
        return ('impute_num', SimpleImputer(strategy=strategy), num_cols)

    def scale_numerical(self, df):
        """
        Returns a StandardScaler for numerical columns.
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return None
        return ('scale_num', StandardScaler(), num_cols)

    def encode_categorical(self, df, handle_unknown='ignore'):
        """
        Returns a OneHotEncoder for categorical columns.
        """
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not cat_cols:
            return None
        return ('encode_cat', OneHotEncoder(handle_unknown=handle_unknown, sparse=False), cat_cols)

    def text_features_processing(self, df, steps=None):
        """
        Returns a ColumnTransformer for text columns using CountVectorizer or TfidfVectorizer.
        """
        text_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].str.len().mean() > 20]
        transformers = []
        if not text_cols:
            return None
        for col in text_cols:
            if steps and 'tfidf' in steps:
                transformers.append((f'tfidf_{col}', TfidfVectorizer(), col))
            else:
                transformers.append((f'count_{col}', CountVectorizer(), col))
        return ('text_features', ColumnTransformer(transformers), text_cols)

    def get_estimator(self, name):
        """
        Instantiates estimator from string name.
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.svm import SVC, SVR
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        estimators = {
            'RandomForestClassifier': RandomForestClassifier(),
            'RandomForestRegressor': RandomForestRegressor(),
            'LogisticRegression': LogisticRegression(),
            'LinearRegression': LinearRegression(),
            'SVC': SVC(),
            'SVR': SVR(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
        }
        return estimators.get(name)

    def create_pipeline(self, algorithm_name, preprocessing_steps, algo_params=None):
        """
        Creates a scikit-learn Pipeline with preprocessing and estimator.
        preprocessing_steps: list of tuples (name, transformer, columns)
        algo_params: dict of estimator parameters
        """
        steps = []
        # Build preprocessing
        transformers = []
        for step in preprocessing_steps:
            if step:
                name, transformer, columns = step
                transformers.append((name, transformer, columns))
        if transformers:
            preprocessor = ColumnTransformer(transformers)
            steps.append(('preprocessor', preprocessor))
        # Estimator
        estimator = self.get_estimator(algorithm_name)
        if estimator is None:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        if algo_params:
            estimator.set_params(**algo_params)
        steps.append(('estimator', estimator))
        return Pipeline(steps)
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