# core/optimizer.py
import time
import logging
from dask_ml.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from core.resource_manager import get_system_info, estimate_training_time

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

    def optimize_hyperparameters(self, pipeline_template, X_train, y_train, X_test, y_test, 
                               search_grid, problem_type, cv=3, use_pareto=False, logger_callback=None):
        """
        Performs HPO using Dask-ML's GridSearchCV.
        A logger_callback can be passed to stream live updates.
        """
        def log(message):
            if logger_callback:
                logger_callback(message)
            logging.info(message)

        system_info = get_system_info()
        log(f"System Info: {system_info['ram']:.2f} GB RAM, {system_info['cpu_cores']} CPU cores.")

        if not search_grid:
            log(f"No search grid. Fitting {pipeline_template.steps[-1][0]} with default parameters.")
            estimated_time = estimate_training_time(pipeline_template.steps[-1][1], len(X_train))
            log(f"Estimated training time: {estimated_time:.2f}s")

            start_time = time.time()
            pipeline_template.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = pipeline_template.predict(X_test).compute()
            y_test_computed = y_test.compute()
            accuracy = accuracy_score(y_test_computed, y_pred)
            f1 = f1_score(y_test_computed, y_pred, average='weighted' if problem_type == 'classification' else 'micro')

            return {
                'best_pipeline': pipeline_template, 'best_params': None, 'best_cv_score': None,
                'test_accuracy': accuracy, 'test_f1_score': f1,
                'hpo_time': 0, 'training_time': train_time
            }

        log(f"Setting up GridSearchCV for {pipeline_template.steps[-1][0]} with {len(search_grid)} parameter combinations.")
        grid_search = GridSearchCV(
            pipeline_template, search_grid, cv=cv, scoring="accuracy", return_train_score=True
        )
        
        log("Starting Dask-ML GridSearchCV...")
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        hpo_time = time.time() - start_time
        log(f"GridSearchCV finished in {hpo_time:.2f}s. Best score: {grid_search.best_score_:.4f}")

        if use_pareto:
            log("Analyzing Pareto front for best accuracy/time trade-off...")
            best_pipeline = self._get_pareto_optimal_model(grid_search)
            log("Selected Pareto-optimal model.")
        else:
            best_pipeline = grid_search.best_estimator_
        
        log("Evaluating final model on the test set...")
        y_pred = best_pipeline.predict(X_test).compute()
        y_test_computed = y_test.compute()
        test_accuracy = accuracy_score(y_test_computed, y_pred)
        test_f1 = f1_score(y_test_computed, y_pred, average='weighted' if problem_type == 'classification' else 'micro')
        log("Evaluation complete.")

        return {
            'best_pipeline': best_pipeline,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1,
            'hpo_time': hpo_time,
            'training_time': hpo_time
        }

    def _get_pareto_optimal_model(self, grid_search):
        """
        Selects the best model from a Pareto front of accuracy and training time.
        """
        cv_results = grid_search.cv_results_
        
        # Create a list of (accuracy, training_time, model_index) tuples
        results = []
        for i, params in enumerate(cv_results['params']):
            accuracy = cv_results['mean_test_score'][i]
            fit_time = cv_results['mean_fit_time'][i]
            results.append((accuracy, fit_time, i))

        # Sort by accuracy (desc) and then by training time (asc)
        results.sort(key=lambda x: (-x[0], x[1]))
        
        # The best model is the first one in the sorted list
        best_index = results[0][2]
        
        # Re-fit the best estimator with the best parameters
        best_params = cv_results['params'][best_index]
        best_pipeline = grid_search.estimator.set_params(**best_params)
        
        return best_pipeline
