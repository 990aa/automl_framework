import logging
import os

def setup_logging():
    """Set up basic logging to a file."""
    log_file = 'automl.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='w'  # Overwrite the log file on each run
    )
    # Add a console handler to also print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info(f"Logging initialized. Log file: {os.path.abspath(log_file)}")

if __name__ == '__main__':
    setup_logging()
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")