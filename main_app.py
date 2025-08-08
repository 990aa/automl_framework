import sys
import logging
import time
from dask.distributed import Client
import dask.dataframe as dd
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from ui.dashboard_widget import DashboardWidget
from ui.progress_widget import ProgressWidget
from ui.comparison_widget import ComparisonWidget
from utils.logger import setup_logging
from utils.session_manager import SessionManager
from core.data_analysis import DataAnalyzer
from core.meta_learning import MetaLearner
from core.model_manager import ModelManager
from core.optimizer import Optimizer
from core.pipeline_builder import PipelineBuilder
from core.worker import AutoMLWorker

from PyQt6.QtCore import pyqtSignal, QThread, QObject


class AutoMLApp(QMainWindow):
    """The main application window for the AutoML framework."""
    progress_updated = pyqtSignal(int, str)

    def __init__(self):
        """Initializes the application, Dask client, and UI."""
        super().__init__()
        try:
            self.client = Client(processes=False) # Dask client
            logging.info(f"Dask dashboard link: {self.client.dashboard_link}")

            self.model_manager = ModelManager()
            self.meta_learner = MetaLearner(experiences_path='data/past_experiences.json')
            self.trained_results = []
            self.dataset_profile = None
            self._init_ui()
        except Exception as e:
            logging.error(f"Error during application initialization: {e}", exc_info=True)
            QMessageBox.critical(self, "Initialization Error", f"An error occurred during initialization:\n{e}")
            sys.exit(1)

    def _init_ui(self):
        """Initializes the main UI components and connects signals."""
        self.setWindowTitle("AutoML Framework (Dask Enabled)")
        self.setGeometry(100, 100, 800, 600)
        self.dashboard = DashboardWidget(self.meta_learner, self.model_manager, self)
        self.setCentralWidget(self.dashboard)
        self.progress_widget = ProgressWidget()
        # Pass model_manager and meta_learner to ComparisonWidget
        self.comparison_widget = ComparisonWidget(self.model_manager, self.meta_learner)
        self.dashboard.start_automl_button.clicked.connect(self.start_automl_process)
        self.dashboard.save_session_button.clicked.connect(self.save_session)
        self.dashboard.load_session_button.clicked.connect(self.load_session)
        self.progress_updated.connect(self.dashboard.update_progress)

    def start_automl_process(self):
        """Starts the AutoML process in a separate thread."""
        try:
            ddf = self.dashboard.dataframe # Now a Dask DataFrame
            dataset_profile = self.dashboard.dataset_profile
            recommendations = self.dashboard.recommendations

            if ddf is None or not dataset_profile or not recommendations:
                QMessageBox.warning(self, "Prerequisites Missing", "Please load a dataset and ensure recommendations are generated before starting.")
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
            self.worker.model_tested.connect(self.dashboard.update_models_tested)
            self.worker.finished.connect(self.on_automl_finished)
            self.worker.progress_updated.connect(self.dashboard.update_progress)

            self.thread.started.connect(self.worker.run)
            self.thread.start()
        except Exception as e:
            logging.error(f"Error starting AutoML process: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"An unexpected error occurred while starting the process:\n{e}")
            self.dashboard.set_ui_enabled(True)

    def on_automl_finished(self):
        """Handles the completion of the AutoML process."""
        self.thread.quit()
        self.thread.wait()
        self.dashboard.set_ui_enabled(True)
        self.progress_widget.hide()
        # Pass both results and the dataset profile to the comparison widget
        self.comparison_widget.display_results(self.trained_results, self.dataset_profile)
        self.comparison_widget.show()

    def save_session(self):
        """Saves the current session to a JSON file."""
        if not self.dataset_profile or not self.trained_results:
            QMessageBox.warning(self, "Cannot Save", "There is no session data to save. Please run the AutoML process first.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Session", "", "JSON Files (*.json)")
        if not save_path:
            return

        # We can't save the pipeline objects directly in JSON, so we remove them
        results_to_save = []
        for res in self.trained_results:
            new_res = res.copy()
            new_res.pop('pipeline', None) # Remove non-serializable pipeline
            results_to_save.append(new_res)

        session_data = {
            "dataset_path": self.dashboard.get_dataset_path(),
            "dataset_profile": self.dataset_profile,
            "trained_results": results_to_save 
        }

        if SessionManager.save_session(save_path, session_data):
            QMessageBox.information(self, "Success", f"Session saved to {save_path}")
        else:
            QMessageBox.critical(self, "Error", "Failed to save the session.")

    def load_session(self):
        """Loads a session from a JSON file."""
        load_path, _ = QFileDialog.getOpenFileName(self, "Load Session", "", "JSON Files (*.json)")
        if not load_path:
            return

        session_data = SessionManager.load_session(load_path)
        if not session_data:
            QMessageBox.critical(self, "Error", "Failed to load the session file.")
            return

        try:
            # Restore dataset
            self.dataset_profile = session_data.get("dataset_profile")
            dataset_path = session_data.get("dataset_path")
            self.dashboard.dataset_path = dataset_path
            self.dashboard.dataset_path_label.setText(dataset_path)
            self.dashboard.dataframe = dd.read_csv(dataset_path, engine='pyarrow')
            
            # Restore UI
            self.dashboard.analyze_and_update_ui()
            
            # Restore results
            self.trained_results = session_data.get("trained_results", [])
            
            # Display results in comparison widget
            self.comparison_widget.display_results(self.trained_results, self.dataset_profile)
            self.comparison_widget.show()
            
            self.dashboard.start_automl_button.setEnabled(True)
            QMessageBox.information(self, "Success", f"Session loaded from {load_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while loading the session data:\n{e}")
            logging.error(f"Failed to process session data from {load_path}: {e}", exc_info=True)

    def closeEvent(self, event):
        """
        Handles the application close event, ensuring the Dask client is shut down.

        Args:
            event: The close event.
        """
        logging.info("Closing Dask client...")
        self.client.close()
        event.accept()


if __name__ == "__main__":
    try:
        setup_logging()
        app = QApplication(sys.argv)
        main_window = AutoMLApp()
        main_window.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.critical(f"A critical error occurred: {e}", exc_info=True)
        # Optionally, show a simple message box if Qt is available
        try:
            app = QApplication(sys.argv)
            QMessageBox.critical(None, "Critical Error", f"A critical error occurred and the application must close:\n{e}")
        finally:
            sys.exit(1)
