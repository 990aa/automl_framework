# ui/comparison_widget.py
import logging
import json
import time
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QLabel, QGroupBox, QHBoxLayout, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

class ComparisonWidget(QWidget):
    """
    A widget to display, compare, and interact with trained models.
    """
    def __init__(self, model_manager, meta_learner, parent=None):
        super().__init__(parent)
        self.results = []
        self.selected_model_pipeline = None
        self.model_manager = model_manager
        self.meta_learner = meta_learner
        self.dataset_profile = None # This will be set when results are displayed
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("Model Comparison Dashboard")
        layout = QVBoxLayout(self)
        # Add logo at the top
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtCore import Qt
    logo_label = QLabel()
    pixmap = QPixmap("logo.jpg")
    logo_label.setPixmap(pixmap.scaledToHeight(48))
    logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(logo_label)

        title = QLabel("AutoML Run Results")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # --- Results Table ---
        table_group = QGroupBox("Model Performance")
        table_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Algorithm", "Accuracy", "F1 Score", "Training Time (s)", "Parameters"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents) # Parameters column
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.results_table.setSortingEnabled(True)

        table_layout.addWidget(self.results_table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        # --- Action Buttons ---
        button_layout = QHBoxLayout()
        self.select_model_button = QPushButton("Confirm Selection")
        self.select_model_button.clicked.connect(self.select_model)
        
        self.save_to_db_button = QPushButton("Save to Meta-DB")
        self.save_to_db_button.clicked.connect(self.save_experience_to_db)
        self.save_to_db_button.setEnabled(False)

        button_layout.addWidget(self.select_model_button)
        button_layout.addWidget(self.save_to_db_button)
        layout.addLayout(button_layout)

        self.resize(900, 500)

    def display_results(self, results_data, dataset_profile):
        """
        Populates the table with results and stores the dataset profile.
        """
        self.results = sorted(results_data, key=lambda x: x.get('accuracy', 0), reverse=True)
        self.dataset_profile = dataset_profile
        self.results_table.setRowCount(len(self.results))
        self.results_table.setSortingEnabled(False)

        for i, result in enumerate(self.results):
            params_str = json.dumps(result.get('best_params'), indent=2, sort_keys=True) if result.get('best_params') else "Default"
            params_str = params_str.replace('model__', '')

            self.results_table.setItem(i, 0, QTableWidgetItem(result.get('algorithm', 'N/A')))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{result.get('accuracy', 0):.4f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{result.get('f1_score', 0):.4f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{result.get('training_time', 0):.2f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(params_str))

        self.results_table.setSortingEnabled(True)
        if self.results:
            self.results_table.selectRow(0)
            self.select_model() # Auto-select the best one initially

    def select_model(self):
        """
        Confirms the selection of a model from the table.
        """
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows:
            self.save_to_db_button.setEnabled(False)
            self.export_model_button.setEnabled(False)
            return

        selected_index = selected_rows[0].row()
        algo_name = self.results_table.item(selected_index, 0).text()
        selected_result = next((r for r in self.results if r['algorithm'] == algo_name), None)

        if selected_result:
            self.selected_model_pipeline = selected_result.get('pipeline')
            self.selected_result_data = selected_result
            logging.info(f"Model selected for action: {selected_result.get('algorithm')}")
            self.save_to_db_button.setEnabled(True)
            self.export_model_button.setEnabled(True)
        else:
            self.save_to_db_button.setEnabled(False)
            self.export_model_button.setEnabled(False)
            QMessageBox.critical(self, "Error", "Could not find the selected model's data.")

    def save_experience_to_db(self):
        """
        Saves the selected model pipeline and adds its metadata to the experience database.
        """
        if not self.selected_result_data or not self.dataset_profile:
            QMessageBox.warning(self, "Action Failed", "No model selected or dataset profile is missing.")
            return

        result = self.selected_result_data
        pipeline = self.selected_model_pipeline
        
        # 1. Save the pipeline object
        pipeline_filename = f"{result['algorithm']}_{int(time.time())}.joblib"
        pipeline_path = self.model_manager.save_pipeline(pipeline, pipeline_filename)
        
        # 2. Create the new experience dictionary
        new_experience = {
            "project_name": f"automl_run_{int(time.time())}",
            "dataset_profile": self.dataset_profile,
            "best_pipeline": {
                "model": result['algorithm'],
                "preprocessing": [step[0] for step in pipeline.steps[:-1]], # Get preproc step names
                "pipeline_filepath": pipeline_path,
                "performance": {
                    "accuracy": result['accuracy'],
                    "f1_score": result['f1_score'],
                    "training_time": result['training_time']
                },
                "hyperparameters": result.get('best_params', {})
            }
        }
        
        # 3. Add to meta-learner
        self.meta_learner.add_experience(new_experience)
        
        QMessageBox.information(
            self, 
            "Experience Saved", 
            f"The selected model '{result['algorithm']}' and its metadata have been saved to the meta-learning database."
        )
        logging.info(f"Saved new experience for model {result['algorithm']} to database.")

    def get_selected_model_pipeline(self):
        """Returns the pipeline of the user-selected model."""
        return self.selected_model_pipeline
