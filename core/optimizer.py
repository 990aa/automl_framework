# core/optimizer.py
import time
import logging
from dask_ml.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

class Optimizer:
    def __init__(self):
        # Grids are defined for Dask-ML compatible models
        self.default_hyperparameter_grids = {
            "RandomForestClassifier": {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
            },
            "LogisticRegression": {
                'C': [0.1, 1.0, 10.0],
            },
        }

    def get_default_grid(self, estimator_name):
        grid = self.default_hyperparameter_grids.get(estimator_name)
        if not grid:
            return None
        return {f"model__{key}": value for key, value in grid.items()}

    def optimize_hyperparameters(self, pipeline_template, X_train, y_train, X_test, y_test, search_grid, problem_type, cv=3):
        """
        Performs HPO using Dask-ML's GridSearchCV.
        """
        if not search_grid:
            # Fallback for models without a grid: just fit and evaluate
            logging.info(f"No search grid. Fitting {pipeline_template.steps[-1][0]} with default parameters.")
            start_time = time.time()
            pipeline_template.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Compute predictions and metrics
            y_pred = pipeline_template.predict(X_test).compute()
            y_test_computed = y_test.compute()
            accuracy = accuracy_score(y_test_computed, y_pred)
            f1 = f1_score(y_test_computed, y_pred, average='weighted' if problem_type == 'classification' else 'micro')

            return {
                'best_pipeline': pipeline_template, 'best_params': None, 'best_cv_score': None,
                'test_accuracy': accuracy, 'test_f1_score': f1,
                'hpo_time': 0, 'training_time': train_time
            }

        # Use Dask-ML's GridSearchCV
        grid_search = GridSearchCV(
            pipeline_template, search_grid, cv=cv, scoring="accuracy"
        )
        
        logging.info(f"Starting Dask-ML GridSearchCV for {pipeline_template.steps[-1][0]}...")
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        hpo_time = time.time() - start_time
        logging.info(f"GridSearchCV finished in {hpo_time:.2f}s. Best score: {grid_search.best_score_:.4f}")

        best_pipeline = grid_search.best_estimator_
        
        # Evaluate the final model on the test set
        y_pred = best_pipeline.predict(X_test).compute()
        y_test_computed = y_test.compute() # Ensure y_test is in memory for scoring
        test_accuracy = accuracy_score(y_test_computed, y_pred)
        test_f1 = f1_score(y_test_computed, y_pred, average='weighted' if problem_type == 'classification' else 'micro')

        return {
            'best_pipeline': best_pipeline,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1,
            'hpo_time': hpo_time,
            'training_time': hpo_time
        }
