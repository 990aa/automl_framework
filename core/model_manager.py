# core/model_manager.py
import time
import os
import joblib
from dask_ml.linear_model import LogisticRegression
from dask_ml.ensemble import RandomForestClassifier

# Note: Not all scikit-learn models have a Dask-ML equivalent.
# We are using a subset here. GradientBoostingClassifier, SVC, etc.,
# would require more specialized handling or might not be available for distributed training.

class ModelManager:
    def __init__(self, model_dir='trained_models'):
        self.models = {
            "RandomForestClassifier": RandomForestClassifier,
            "LogisticRegression": LogisticRegression,
            # Add other Dask-ML compatible models here
        }
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def get_estimator(self, name):
        """Instantiates a Dask-ML estimator from a string name."""
        model_class = self.models.get(name)
        if model_class:
            # Dask-ML models are initialized similarly to scikit-learn
            return model_class()
        
        # Fallback for models not in Dask-ML but compatible via joblib backend
        try:
            from sklearn.tree import DecisionTreeClassifier
            if name == 'DecisionTreeClassifier':
                return DecisionTreeClassifier()
        except ImportError:
            pass

        raise ValueError(f"Model '{name}' not found in Dask-ML model manager.")

    def list_models(self):
        return list(self.models.keys())

    def train_model(self, pipeline, X_train, y_train):
        """
        Trains the Dask-ML compatible pipeline.
        This is a lazy operation; actual computation happens on predict or score.
        """
        start_time = time.time()
        # The fit call is still lazy for many Dask-ML components
        pipeline.fit(X_train, y_train)
        end_time = time.time()
        # This time measures the time to construct the graph, not compute it
        training_time = end_time - start_time
        return pipeline, training_time

    def save_pipeline(self, pipeline, filename):
        """Saves a trained pipeline to a file."""
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump(pipeline, filepath)
        return filepath
