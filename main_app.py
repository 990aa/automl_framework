import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from ui.dashboard_widget import DashboardWidget
from utils.logger import setup_logging

class AutoMLApp(QMainWindow):
    """The main application window."""
    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        """Initialize the main window UI."""
        self.setWindowTitle("AutoML Framework")
        self.setGeometry(100, 100, 800, 600)  # x, y, width, height

        # Set the central widget
        self.dashboard = DashboardWidget(self)
        self.setCentralWidget(self.dashboard)

        # --- Signal Connections ---
        # self.dashboard.dataset_loaded.connect(self.handle_dataset_load)

    # def handle_dataset_load(self, file_path):
    #     """Placeholder to handle the dataset loaded signal."""
    #     logging.info(f"Main app received dataset path: {file_path}")
    #     # Here you would trigger the core logic (e.g., data analysis)
    #     pass

if __name__ == "__main__":
    # Set up logging first
    setup_logging()

    # Create and run the application
    app = QApplication(sys.argv)
    main_window = AutoMLApp()
    main_window.show()
    sys.exit(app.exec())