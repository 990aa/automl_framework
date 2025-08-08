# ui/progress_widget.py

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QProgressBar, QTextEdit, QLabel
from PyQt6.QtCore import pyqtSlot, Qt
from PyQt6.QtGui import QPixmap

class ProgressWidget(QWidget):
    """A widget to display AutoML progress with a progress bar and log."""

    def __init__(self, parent=None):
        """
        Initializes the ProgressWidget.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("AutoML Progress")
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        # Add logo at the top
        from PyQt6.QtGui import QPixmap
        from PyQt6.QtCore import Qt
        logo_label = QLabel()
        pixmap = QPixmap("logo.jpg")
        logo_label.setPixmap(pixmap.scaledToHeight(48))
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_label)

        self.progress_label = QLabel("Overall Progress:")
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.log_label = QLabel("Details:")
        layout.addWidget(self.log_label)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        self.setLayout(layout)
        self.resize(500, 400)

    @pyqtSlot(int, str)
    def update_progress(self, value, message):
        """
        Update the overall progress bar and its label.

        Args:
            value (int): The progress value (0-100).
            message (str): The message to display next to the progress bar.
        """
        self.progress_bar.setValue(value)
        # The message from the main app's signal is more for the dashboard,
        # so we'll just show a generic percentage here.
        self.progress_label.setText(f"Overall Progress: {message}")


    @pyqtSlot(str)
    def add_log_message(self, message):
        """
        Append a message to the log area.

        Args:
            message (str): The message to append to the log.
        """
        self.log_area.append(message)
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def clear_logs(self):
        """Clears the log area."""
        self.log_area.clear()