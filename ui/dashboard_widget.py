import pandas as pd
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QProgressBar, QTableWidget, QTableWidgetItem, QGroupBox, QFormLayout
)
from PyQt6.QtCore import pyqtSignal
from core.data_analysis import DataAnalyzer

class DashboardWidget(QWidget):
    """Main dashboard widget for the AutoML application."""
    dataset_loaded = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_path = None
        self.dataframe = None
        self.dataset_profile = None  # To store the analysis results
        self.core_analyzer = DataAnalyzer()
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        main_layout = QVBoxLayout(self)

        # --- Title ---
        title_label = QLabel("AutoML Dashboard")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(title_label)

        # --- File Operations Section ---
        file_ops_group = QGroupBox("File Operations")
        file_ops_layout = QVBoxLayout()

        self.upload_button = QPushButton("Upload Dataset (.csv)")
        self.upload_button.clicked.connect(self.upload_dataset)
        file_ops_layout.addWidget(self.upload_button)

        self.load_project_button = QPushButton("Load Previous Project")
        self.load_project_button.setEnabled(False) # For later implementation
        file_ops_layout.addWidget(self.load_project_button)

        self.dataset_path_label = QLabel("No dataset loaded.")
        self.dataset_path_label.setWordWrap(True)
        file_ops_layout.addWidget(self.dataset_path_label)

        file_ops_group.setLayout(file_ops_layout)
        main_layout.addWidget(file_ops_group)

        # --- Dataset Info Section ---
        dataset_info_group = QGroupBox("Dataset Info")
        dataset_info_layout = QFormLayout()

        self.rows_label = QLabel("N/A")
        self.features_label = QLabel("N/A")
        self.problem_type_label = QLabel("N/A")
        self.similarity_label = QLabel("N/A") # For meta-learning in a future phase

        dataset_info_layout.addRow("Rows:", self.rows_label)
        dataset_info_layout.addRow("Features:", self.features_label)
        dataset_info_layout.addRow("Problem Type:", self.problem_type_label)
        dataset_info_layout.addRow("Similarity Score:", self.similarity_label)

        dataset_info_group.setLayout(dataset_info_layout)
        main_layout.addWidget(dataset_info_group)

        # --- AutoML Control Section ---
        self.start_automl_button = QPushButton("Start AutoML Process")
        self.start_automl_button.clicked.connect(self.start_automl_process)
        self.start_automl_button.setEnabled(False) # Disabled until dataset is loaded
        self.start_automl_button.setStyleSheet("font-size: 16px; padding: 10px;")
        main_layout.addWidget(self.start_automl_button)

        main_layout.addStretch() # Pushes everything to the top

    def upload_dataset(self):
        """Open a file dialog, load data, and trigger analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Dataset File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.dataset_path = file_path
            self.dataset_path_label.setText(f"Loaded: {self.dataset_path}")
            logging.info(f"Dataset selected: {self.dataset_path}")
            
            try:
                import pandas as pd
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QProgressBar, QTableWidget, QTableWidgetItem, QGroupBox, QFormLayout
)
from PyQt6.QtCore import pyqtSignal
from core.data_analysis import DataAnalyzer
from core.meta_learning import MetaLearner

class DashboardWidget(QWidget):
    """Main dashboard widget for the AutoML application."""
    dataset_loaded = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_path = None
        self.dataframe = None
        self.dataset_profile = None
        self.recommendations = [] # Store recommendations
        self.core_analyzer = DataAnalyzer()
        self.meta_learner = MetaLearner()
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        main_layout = QVBoxLayout(self)

        # --- Title ---
        title_label = QLabel("AutoML Dashboard")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(title_label)

        # --- File Operations Section ---
        file_ops_group = QGroupBox("File Operations")
        file_ops_layout = QVBoxLayout()

        self.upload_button = QPushButton("Upload Dataset (.csv)")
        self.upload_button.clicked.connect(self.upload_dataset)
        file_ops_layout.addWidget(self.upload_button)

        self.load_project_button = QPushButton("Load Previous Project")
        self.load_project_button.setEnabled(False) # For later implementation
        file_ops_layout.addWidget(self.load_project_button)

        self.dataset_path_label = QLabel("No dataset loaded.")
        self.dataset_path_label.setWordWrap(True)
        file_ops_layout.addWidget(self.dataset_path_label)

        file_ops_group.setLayout(file_ops_layout)
        main_layout.addWidget(file_ops_group)

        # --- Dataset Info Section ---
        dataset_info_group = QGroupBox("Dataset Info")
        dataset_info_layout = QFormLayout()

        self.rows_label = QLabel("N/A")
        self.features_label = QLabel("N/A")
        self.problem_type_label = QLabel("N/A")
        self.similarity_label = QLabel("N/A")

        dataset_info_layout.addRow("Rows:", self.rows_label)
        dataset_info_layout.addRow("Features:", self.features_label)
        dataset_info_layout.addRow("Problem Type:", self.problem_type_label)
        dataset_info_layout.addRow("Most Similar Project:", self.similarity_label)

        dataset_info_group.setLayout(dataset_info_layout)
        main_layout.addWidget(dataset_info_group)

        # --- Recommendations Section ---
        reco_group = QGroupBox("Algorithm Recommendations")
        reco_layout = QVBoxLayout()
        self.reco_table = QTableWidget()
        self.reco_table.setColumnCount(4)
        self.reco_table.setHorizontalHeaderLabels(["Rank", "Model", "Preprocessing", "Source Project(s)"])
        self.reco_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        reco_layout.addWidget(self.reco_table)
        reco_group.setLayout(reco_layout)
        main_layout.addWidget(reco_group)

        # --- AutoML Control Section ---
        self.start_automl_button = QPushButton("Start AutoML Process")
        self.start_automl_button.clicked.connect(self.start_automl_process)
        self.start_automl_button.setEnabled(False) # Disabled until dataset is loaded
        self.start_automl_button.setStyleSheet("font-size: 16px; padding: 10px;")
        main_layout.addWidget(self.start_automl_button)

        main_layout.addStretch() # Pushes everything to the top

    def upload_dataset(self):
        """Open a file dialog, load data, and trigger analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Dataset File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.dataset_path = file_path
            self.dataset_path_label.setText(f"Loaded: {self.dataset_path}")
            logging.info(f"Dataset selected: {self.dataset_path}")
            
            try:
                self.dataframe = pd.read_csv(file_path)
                logging.info("CSV file loaded into DataFrame successfully.")
                self.analyze_and_update_ui()
                self.start_automl_button.setEnabled(True)
                self.dataset_loaded.emit(self.dataset_path)
            except Exception as e:
                logging.error(f"Failed to load or analyze CSV: {e}")
                self.dataset_path_label.setText(f"Error: Could not load file. See log for details.")
                self.start_automl_button.setEnabled(False)


    def analyze_and_update_ui(self):
        """Run the data analyzer and update the UI with the results."""
        if self.dataframe is None:
            return
        
        logging.info("Starting data analysis...")
        self.dataset_profile = self.core_analyzer.analyze_dataset(self.dataframe)
        
        # Update UI labels with analysis results
        self.rows_label.setText(str(self.dataset_profile.get('n_rows', 'N/A')))
        self.features_label.setText(str(self.dataset_profile.get('n_features', 'N/A')))
        self.problem_type_label.setText(str(self.dataset_profile.get('problem_type', 'Unknown')))
        
        logging.info("UI updated with dataset analysis results.")
        
        # Now, get meta-learning recommendations
        self.update_recommendations()

    def update_recommendations(self):
        """Use the meta-learner to find and display recommendations."""
        if not self.dataset_profile:
            return
        
        # Get ranked list of recommendations
        self.recommendations = self.meta_learner.recommend_algorithms(self.dataset_profile)
        
        # Update the summary label
        if self.recommendations:
            # Count unique source projects from the top recommendations
            source_projects = set()
            for reco in self.recommendations:
                source_projects.update(reco['source_projects'])
            self.similarity_label.setText(f"Similar to {len(source_projects)} past project(s)")
        else:
            self.similarity_label.setText("No similar projects found.")

        # Update the recommendations table
        self.reco_table.setRowCount(len(self.recommendations))
        for i, reco in enumerate(self.recommendations):
            self.reco_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.reco_table.setItem(i, 1, QTableWidgetItem(reco['model']))
            self.reco_table.setItem(i, 2, QTableWidgetItem(", ".join(reco['preprocessing'])))
            self.reco_table.setItem(i, 3, QTableWidgetItem(", ".join(reco['source_projects'])))
        
        self.reco_table.resizeColumnsToContents()


    def start_automl_process(self):
        """Placeholder for starting the AutoML process."""
        logging.info("Starting AutoML process...")
        # This will be connected to the core logic later
        self.start_automl_button.setEnabled(False)
        self.upload_button.setEnabled(False)

                logging.info("CSV file loaded into DataFrame successfully.")
                self.analyze_and_update_ui()
                self.start_automl_button.setEnabled(True)
                self.dataset_loaded.emit(self.dataset_path)
            except Exception as e:
                logging.error(f"Failed to load or analyze CSV: {e}")
                self.dataset_path_label.setText(f"Error: Could not load file. See log for details.")
                self.start_automl_button.setEnabled(False)


    def analyze_and_update_ui(self):
        """Run the data analyzer and update the UI with the results."""
        if self.dataframe is None:
            return
        
        logging.info("Starting data analysis...")
        self.dataset_profile = self.core_analyzer.analyze_dataset(self.dataframe)
        
        # Update UI labels
        self.rows_label.setText(str(self.dataset_profile.get('n_rows', 'N/A')))
        self.features_label.setText(str(self.dataset_profile.get('n_features', 'N/A')))
        self.problem_type_label.setText(str(self.dataset_profile.get('problem_type', 'Unknown')))
        
        logging.info("UI updated with dataset analysis results.")
        # The similarity score will be updated in a later phase.

    def start_automl_process(self):
        """Placeholder for starting the AutoML process."""
        logging.info("Starting AutoML process...")
        # This will be connected to the core logic later
        self.start_automl_button.setEnabled(False)
        self.upload_button.setEnabled(False)