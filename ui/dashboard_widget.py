import dask.dataframe as dd
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QProgressBar, QTableWidget, QTableWidgetItem, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal
from core.data_analysis import DataAnalyzer

class DashboardWidget(QWidget):
    """Main dashboard widget for the AutoML application."""
    dataset_loaded = pyqtSignal(str)

    def __init__(self, meta_learner, model_manager, parent=None):
        super().__init__(parent)
        self.dataset_path = None
        self.dataframe = None # This will be a Dask DataFrame
        self.dataset_profile = None
        self.recommendations = []
        self.core_analyzer = DataAnalyzer()
        self.meta_learner = meta_learner
        self.model_manager = model_manager
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        main_layout = QVBoxLayout(self)
        # Add logo at the top
        from PyQt6.QtGui import QPixmap
        logo_label = QLabel()
        pixmap = QPixmap("logo.jpg")
        logo_label.setPixmap(pixmap.scaledToHeight(64))
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(logo_label)
        title_label = QLabel("AutoML Dashboard (Dask Enabled)")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(title_label)

        # ... (rest of the UI setup is the same)
        file_ops_group = QGroupBox("File Operations")
        file_ops_layout = QVBoxLayout()
        self.upload_button = QPushButton("Upload Dataset (.csv)")
        self.upload_button.clicked.connect(self.upload_dataset)
        self.load_session_button = QPushButton("Load Session")
        self.save_session_button = QPushButton("Save Session")
        file_ops_layout.addWidget(self.upload_button)
        file_ops_layout.addWidget(self.load_session_button)
        file_ops_layout.addWidget(self.save_session_button)
        self.dataset_path_label = QLabel("No dataset loaded.")
        file_ops_layout.addWidget(self.dataset_path_label)
        file_ops_group.setLayout(file_ops_layout)
        main_layout.addWidget(file_ops_group)

        dataset_info_group = QGroupBox("Dataset Info")
        dataset_info_layout = QFormLayout()
        self.rows_label = QLabel("N/A")
        self.features_label = QLabel("N/A")
        self.problem_type_label = QLabel("N/A")
        self.similarity_label = QLabel("N/A")
        dataset_info_layout.addRow("Rows:", self.rows_label)
        dataset_info_layout.addRow("Features:", self.features_label)
        dataset_info_layout.addRow("Problem Type:", self.problem_type_label)
        dataset_info_layout.addRow("Similarity Score:", self.similarity_label)
        dataset_info_group.setLayout(dataset_info_layout)
        main_layout.addWidget(dataset_info_group)

        reco_group = QGroupBox("Algorithm Recommendations")
        reco_layout = QVBoxLayout()
        self.reco_table = QTableWidget()
        self.reco_table.setColumnCount(4)
        self.reco_table.setHorizontalHeaderLabels(["Rank", "Algorithm", "Preprocessing", "Source Projects"])
        reco_layout.addWidget(self.reco_table)
        reco_group.setLayout(reco_layout)
        main_layout.addWidget(reco_group)

        self.start_automl_button = QPushButton("Start AutoML Process")
        self.start_automl_button.setEnabled(False)
        self.start_automl_button.setStyleSheet("font-size: 16px; padding: 10px;")
        main_layout.addWidget(self.start_automl_button)

        progress_group = QGroupBox("Workflow Status")
        progress_layout = QFormLayout()
        self.status_label = QLabel("Idle")
        self.progress_label = QLabel("Waiting to start...")
        self.models_tested_label = QLabel("0")
        self.progress_bar = QProgressBar()
        
        progress_layout.addRow("Status:", self.status_label)
        progress_layout.addRow("Progress:", self.progress_label)
        progress_layout.addRow("Models Tested:", self.models_tested_label)
        progress_layout.addRow(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        main_layout.addWidget(progress_group)
        main_layout.addStretch()


    def upload_dataset(self):
        """Open a file dialog, load data into a Dask DataFrame, and trigger analysis."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Dataset File", "", "CSV Files (*.csv)")
        if file_path:
            self.dataset_path = file_path
            self.update_status(f"Loading: {self.dataset_path}")
            try:
                self.dataframe = dd.read_csv(file_path, engine='pyarrow')
                logging.info(f"Dataset loaded into Dask DataFrame. Partitions: {self.dataframe.npartitions}")
                self.update_status("Analyzing dataset...")
                self.analyze_and_update_ui()
                self.start_automl_button.setEnabled(True)
                self.dataset_loaded.emit(self.dataset_path)
                self.update_status("Ready")
            except Exception as e:
                logging.error(f"Failed to load dataset with Dask: {e}")
                self.update_status(f"Error: Could not load file.")
                self.start_automl_button.setEnabled(False)

    def analyze_and_update_ui(self):
        """Run the data analyzer on the Dask DataFrame and update the UI."""
        try:
            if self.dataframe is None: return
            logging.info("Starting data analysis with Dask...")
            
            self.dataset_profile = self.core_analyzer.analyze_dataset(self.dataframe)
            
            self.rows_label.setText(str(self.dataset_profile.get('n_rows', 'N/A')))
            self.features_label.setText(str(self.dataset_profile.get('n_features', 'N/A')))
            self.problem_type_label.setText(str(self.dataset_profile.get('problem_type', 'Unknown')))
            logging.info("UI updated with dataset analysis.")
            self.update_recommendations()
        except Exception as e:
            logging.error(f"Error during data analysis: {e}", exc_info=True)
            self.update_status(f"Error: Analysis failed.")
            # Optionally show a message box
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Analysis Error", f"An error occurred during dataset analysis:\n{e}")

    def update_recommendations(self):
        try:
            if not self.dataset_profile: return
            self.update_status("Generating recommendations...")
            self.recommendations = self.meta_learner.recommend_algorithms(self.dataset_profile)
            if self.recommendations:
                source_projects = set(p for r in self.recommendations for p in r['source_projects'])
                self.similarity_label.setText(f"Similar to {len(source_projects)} past project(s)")
            else:
                self.similarity_label.setText("No similar projects found.")
            self.reco_table.setRowCount(len(self.recommendations))
            for i, reco in enumerate(self.recommendations):
                self.reco_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                self.reco_table.setItem(i, 1, QTableWidgetItem(reco['model']))
                self.reco_table.setItem(i, 2, QTableWidgetItem(", ".join(reco['preprocessing'])))
                self.reco_table.setItem(i, 3, QTableWidgetItem(", ".join(reco['source_projects'])))
            self.reco_table.resizeColumnsToContents()
            self.update_status("Ready")
        except Exception as e:
            logging.error(f"Error updating recommendations: {e}", exc_info=True)
            self.update_status(f"Error: Recommendation failed.")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Recommendation Error", f"An error occurred while generating recommendations:\n{e}")

    def update_status(self, message):
        """
        Updates the status label with a new message.

        Args:
            message (str): The message to display in the status label.
        """
        self.status_label.setText(message)

    def update_progress(self, value, message):
        """
        Updates the progress bar and its corresponding label.

        Args:
            value (int): The progress value (0-100).
            message (str): The message to display next to the progress bar.
        """
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        if value == 100:
            self.progress_label.setText("Workflow complete.")
            self.update_status("Finished")

    def set_ui_enabled(self, enabled):
        """
        Enables or disables the main UI controls.

        Args:
            enabled (bool): True to enable the controls, False to disable them.
        """
        self.upload_button.setEnabled(enabled)
        self.start_automl_button.setEnabled(enabled)
        if not enabled:
            self.update_status("Running AutoML...")
        else:
            self.update_status("Idle")

    def update_models_tested(self, count):
        """
        Updates the label showing the number of models tested.

        Args:
            count (int): The number of models tested so far.
        """
        self.models_tested_label.setText(str(count))