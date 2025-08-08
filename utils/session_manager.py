# utils/session_manager.py
import json
import logging

class SessionManager:
    @staticmethod
    def save_session(file_path, session_data):
        """Saves the AutoML session to a JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(session_data, f, indent=4)
            logging.info(f"Session saved successfully to {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving session to {file_path}: {e}")
            return False

    @staticmethod
    def load_session(file_path):
        """Loads an AutoML session from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                session_data = json.load(f)
            logging.info(f"Session loaded successfully from {file_path}")
            return session_data
        except Exception as e:
            logging.error(f"Error loading session from {file_path}: {e}")
            return None
