# ui/comparison_widget.py
import logging
import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QLabel, QGroupBox, QHBoxLayout, QMessageBox
)
from PyQt6.QtCore import Qt

class ComparisonWidget(QWidget):
    """
    A widget to display, compare, and interact with trained models.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results = []
        self.selected_model_pipeline = None
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("Model Comparison Dashboard")
        layout = QVBoxLayout(self)

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
        self.select_model_button = QPushButton("Select Model")
        self.select_model_button.clicked.connect(self.select_model)
        
        self.export_model_button = QPushButton("Export Model")
        self.export_model_button.setEnabled(False) # Placeholder
        
        self.deploy_button = QPushButton("Deploy")
        self.deploy_button.setEnabled(False) # Placeholder

        button_layout.addWidget(self.select_model_button)
        button_layout.addWidget(self.export_model_button)
        button_layout.addWidget(self.deploy_button)
        layout.addLayout(button_layout)

        self.resize(900, 500)

    def display_results(self, results_data):
        """
        Populates the table with results from the AutoML run.
        """
        self.results = sorted(results_data, key=lambda x: x.get('accuracy', 0), reverse=True)
        self.results_table.setRowCount(len(self.results))
        self.results_table.setSortingEnabled(False)

        for i, result in enumerate(self.results):
            params_str = json.dumps(result.get('best_params'), indent=2, sort_keys=True) if result.get('best_params') else "Default"
            # Clean up the param string for display
            params_str = params_str.replace('model__', '')

            self.results_table.setItem(i, 0, QTableWidgetItem(result.get('algorithm', 'N/A')))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{result.get('accuracy', 0):.4f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{result.get('f1_score', 0):.4f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{result.get('training_time', 0):.2f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(params_str))

        self.results_table.setSortingEnabled(True)
        if self.results:
            self.results_table.selectRow(0) # Select the best model by default

    def select_model(self):
        """
        Confirms the selection of a model from the table.
        """
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Model Selected", "Please select a model from the table.")
            return

        selected_index = selected_rows[0].row()
        
        # Find the original result dict, as the table might be sorted
        algo_name = self.results_table.item(selected_index, 0).text()
        selected_result = next((r for r in self.results if r['algorithm'] == algo_name), None)

        if selected_result:
            self.selected_model_pipeline = selected_result.get('pipeline')
            logging.info(f"Model selected: {selected_result.get('algorithm')}")
            QMessageBox.information(
                self,
                "Model Selected",
                f"You have selected the {selected_result.get('algorithm')} model.\n"
                f"It is now ready for export or deployment (functionality to be implemented)."
            )
            self.export_model_button.setEnabled(True) # Enable for next step
        else:
            QMessageBox.critical(self, "Error", "Could not find the selected model's data.")

    def get_selected_model_pipeline(self):
        """Returns the pipeline of the user-selected model."""
        return self.selected_model_pipeline
