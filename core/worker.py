# core/worker.py
import logging
from PyQt6.QtCore import pyqtSignal, QObject
from core.pipeline_builder import PipelineBuilder
from core.optimizer import Optimizer

class AutoMLWorker(QObject):
    """
    A QObject worker for running the AutoML process in a separate thread.
    """
    finished = pyqtSignal()
    progress_updated = pyqtSignal(int, str)
    status_updated = pyqtSignal(str)
    log_message_updated = pyqtSignal(str)
    model_tested = pyqtSignal(int)

    def __init__(self, app_instance, ddf, dataset_profile, recommendations):
        """
        Initializes the AutoML worker.

        Args:
            app_instance (AutoMLApp): The main application instance.
            ddf (dask.dataframe.DataFrame): The Dask DataFrame containing the dataset.
            dataset_profile (dict): A dictionary containing the dataset profile.
            recommendations (list): A list of recommended algorithms.
        """
        super().__init__()
        self.app = app_instance
        self.ddf = ddf
        self.dataset_profile = dataset_profile
        self.recommendations = recommendations

    def run(self):
        """
        Executes the main AutoML workflow, including data preparation,
        hyperparameter optimization, and model evaluation.
        """
        try:
            self.status_updated.emit("Initializing workflow...")
            self.log_message_updated.emit("Initializing Dask-based workflow...")
            builder = PipelineBuilder(self.recommendations, self.app.model_manager)
            optimizer = Optimizer()
            
            self.status_updated.emit("Preparing data...")
            self.log_message_updated.emit("Preparing data with Dask...")
            target_column = self.dataset_profile.get('target_column')
            X = self.ddf.drop(columns=[target_column])
            y = self.ddf[target_column]
            
            from dask_ml.model_selection import train_test_split as dask_train_test_split
            X_train, X_test, y_train, y_test = dask_train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )
            self.log_message_updated.emit(f"Data split into training and testing sets.")

            results = []
            num_recommendations = len(self.recommendations)
            self.status_updated.emit("Running Hyperparameter Optimization...")
            for i, reco in enumerate(self.recommendations):
                algo_name = reco['model']
                progress_percent = int(((i + 1) / num_recommendations) * 100)
                progress_message = f"Training {i+1}/{num_recommendations}: {algo_name}"
                self.progress_updated.emit(progress_percent, progress_message)
                self.log_message_updated.emit(f"--- Starting model: {algo_name} ---")
                self.model_tested.emit(i + 1)

                pipeline_template = builder.create_pipeline_from_profile(algo_name, self.dataset_profile, X_train)
                search_grid = optimizer.get_default_grid(algo_name)
                
                hpo_results = optimizer.optimize_hyperparameters(
                    pipeline_template, X_train, y_train, X_test, y_test,
                    search_grid, self.dataset_profile.get('problem_type'),
                    use_pareto=True,
                    logger_callback=self.log_message_updated.emit
                )

                self.log_message_updated.emit(f"HPO/Training completed in {hpo_results['hpo_time']:.2f}s.")
                if hpo_results['best_params']:
                    self.log_message_updated.emit(f"Best CV Score: {hpo_results['best_cv_score']:.4f}")
                self.log_message_updated.emit(f"Final Test Accuracy: {hpo_results['test_accuracy']:.4f}")

                results.append({
                    'algorithm': algo_name,
                    'pipeline': hpo_results['best_pipeline'],
                    'accuracy': hpo_results['test_accuracy'],
                    'f1_score': hpo_results['test_f1_score'],
                    'training_time': hpo_results['training_time'],
                    'hpo_time': hpo_results['hpo_time'],
                    'best_params': hpo_results['best_params']
                })

            self.status_updated.emit("Finalizing results...")
            if results:
                best_result = max(results, key=lambda x: x['accuracy'])
                self.log_message_updated.emit(f"\n--- Best Overall Model Found: {best_result['algorithm']} ---")
                self.log_message_updated.emit(f"Accuracy: {best_result['accuracy']:.4f}, F1-Score: {best_result['f1_score']:.4f}")
                self.log_message_updated.emit("You can now compare all models and save the best one to the meta-learning database.")

            self.app.trained_results = results
            self.app.dataset_profile = self.dataset_profile
            self.log_message_updated.emit("\n--- Workflow complete ---")
            self.progress_updated.emit(100, "Workflow complete!")

        except Exception as e:
            logging.error(f"An error occurred during the AutoML workflow: {e}", exc_info=True)
            self.log_message_updated.emit(f"Error: {e}")
            self.status_updated.emit(f"Error!")
        finally:
            self.finished.emit()
