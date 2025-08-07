import sys
import logging
import time
from dask.distributed import Client

from PyQt6.QtWidgets import QApplication, QMainWindow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from ui.dashboard_widget import DashboardWidget
from ui.progress_widget import ProgressWidget
from ui.comparison_widget import ComparisonWidget
from utils.logger import setup_logging
from core.data_analysis import DataAnalyzer
from core.meta_learning import MetaLearner
from core.model_manager import ModelManager
from core.optimizer import Optimizer
from core.pipeline_builder import PipelineBuilder

from PyQt6.QtCore import pyqtSignal, QThread, QObject

class AutoMLWorker(QObject):
    finished = pyqtSignal()
    progress_updated = pyqtSignal(int, str)
    status_updated = pyqtSignal(str)
    log_message_updated = pyqtSignal(str)

    def __init__(self, app_instance, ddf, dataset_profile, recommendations):
        super().__init__()
        self.app = app_instance
        self.ddf = ddf
        self.dataset_profile = dataset_profile
        self.recommendations = recommendations

    def run(self):
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


class AutoMLApp(QMainWindow):
    progress_updated = pyqtSignal(int, str)

    def __init__(self):
        super().__init__()
        self.client = Client(processes=False) # Dask client
        logging.info(f"Dask dashboard link: {self.client.dashboard_link}")

        self.model_manager = ModelManager()
        self.meta_learner = MetaLearner(experiences_path='data/past_experiences.json')
        self.trained_results = []
        self.dataset_profile = None
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("AutoML Framework (Dask Enabled)")
        self.setGeometry(100, 100, 800, 600)
        self.dashboard = DashboardWidget(self.meta_learner, self.model_manager, self)
        self.setCentralWidget(self.dashboard)
        self.progress_widget = ProgressWidget()
        # Pass model_manager and meta_learner to ComparisonWidget
        self.comparison_widget = ComparisonWidget(self.model_manager, self.meta_learner)
        self.dashboard.start_automl_button.clicked.connect(self.start_automl_process)
        self.progress_updated.connect(self.dashboard.update_progress)

    def start_automl_process(self):
        ddf = self.dashboard.dataframe # Now a Dask DataFrame
        dataset_profile = self.dashboard.dataset_profile
        recommendations = self.dashboard.recommendations

        if ddf is None or not dataset_profile or not recommendations:
            return

        self.progress_widget.clear_logs()
        self.progress_widget.show()
        self.dashboard.set_ui_enabled(False)

        self.thread = QThread()
        self.worker = AutoMLWorker(self, ddf, dataset_profile, recommendations)
        self.worker.moveToThread(self.thread)

        # Connect all worker signals to the appropriate slots
        self.worker.status_updated.connect(self.dashboard.update_status)
        self.worker.progress_updated.connect(self.progress_widget.update_progress)
        self.worker.log_message_updated.connect(self.progress_widget.add_log_message)
        self.worker.finished.connect(self.on_automl_finished)
        self.worker.progress_updated.connect(self.dashboard.update_progress)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def on_automl_finished(self):
        self.thread.quit()
        self.thread.wait()
        self.dashboard.set_ui_enabled(True)
        self.progress_widget.hide()
        # Pass both results and the dataset profile to the comparison widget
        self.comparison_widget.display_results(self.trained_results, self.dataset_profile)
        self.comparison_widget.show()

    def closeEvent(self, event):
        """Shut down the Dask client when the application closes."""
        logging.info("Closing Dask client...")
        self.client.close()
        event.accept()


if __name__ == "__main__":
    setup_logging()
    app = QApplication(sys.argv)
    main_window = AutoMLApp()
    main_window.show()
    sys.exit(app.exec())
