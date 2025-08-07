import sys
import logging
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
    log_message_updated = pyqtSignal(str)

    def __init__(self, app_instance, ddf, dataset_profile, recommendations):
        super().__init__()
        self.app = app_instance
        self.ddf = ddf
        self.dataset_profile = dataset_profile
        self.recommendations = recommendations

    def run(self):
        try:
            self.log_message_updated.emit("Initializing Dask-based workflow...")
            builder = PipelineBuilder(self.recommendations, self.app.model_manager)
            optimizer = Optimizer()
            
            self.log_message_updated.emit("Preparing data with Dask...")
            target_column = self.dataset_profile.get('target_column')
            X = self.ddf.drop(columns=[target_column])
            y = self.ddf[target_column]
            
            # Dask-ML train_test_split
            from dask_ml.model_selection import train_test_split as dask_train_test_split
            X_train, X_test, y_train, y_test = dask_train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )
            self.log_message_updated.emit(f"Data split into training and testing sets.")

            results = []
            num_recommendations = len(self.recommendations)
            for i, reco in enumerate(self.recommendations):
                algo_name = reco['model']
                progress_percent = int(((i + 1) / num_recommendations) * 100)
                self.progress_updated.emit(progress_percent, f"Processing {i+1}/{num_recommendations}: {algo_name}")
                self.log_message_updated.emit(f"--- Starting model: {algo_name} ---")

                pipeline_template = builder.create_pipeline_from_profile(algo_name, self.dataset_profile, X_train)
                search_grid = optimizer.get_default_grid(algo_name)
                
                hpo_results = optimizer.optimize_hyperparameters(
                    pipeline_template, X_train, y_train, X_test, y_test,
                    search_grid, self.dataset_profile.get('problem_type')
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

            self.app.trained_results = results
            self.log_message_updated.emit("\n--- Workflow complete ---")
            self.progress_updated.emit(100, "Workflow complete!")

        except Exception as e:
            logging.error(f"An error occurred during the AutoML workflow: {e}", exc_info=True)
            self.log_message_updated.emit(f"Error: {e}")
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
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("AutoML Framework (Dask Enabled)")
        self.setGeometry(100, 100, 800, 600)
        self.dashboard = DashboardWidget(self.meta_learner, self.model_manager, self)
        self.setCentralWidget(self.dashboard)
        self.progress_widget = ProgressWidget()
        self.comparison_widget = ComparisonWidget()
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
        self.comparison_widget.display_results(self.trained_results)
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
