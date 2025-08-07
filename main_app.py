import sys
import logging
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ui.dashboard_widget import DashboardWidget
from utils.logger import setup_logging
from core.data_analysis import DataAnalyzer
from core.meta_learning import MetaLearner
from core.model_manager import ModelManager
from core.pipeline_builder import PipelineBuilder, handle_missing, scale_numerical, encode_categorical

from PyQt6.QtCore import pyqtSignal

class AutoMLApp(QMainWindow):
    """The main application window."""
    progress_updated = pyqtSignal(int, str)

    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
        self.meta_learner = MetaLearner(experiences_path='data/past_experiences.json')
        self._init_ui()

    def _init_ui(self):
        """Initialize the main window UI."""
        self.setWindowTitle("AutoML Framework")
        self.setGeometry(100, 100, 800, 600)  # x, y, width, height

        # Set the central widget
        self.dashboard = DashboardWidget(self.meta_learner, self.model_manager, self)
        self.setCentralWidget(self.dashboard)

        # --- Signal Connections ---
        self.dashboard.start_automl_button.clicked.connect(self.start_automl_process)
        self.progress_updated.connect(self.dashboard.update_progress)

    def start_automl_process(self):
        """
        Slot to start the AutoML workflow.
        """
        self.dashboard.progress_bar.setVisible(True)
        self.run_automl_workflow(self.dashboard.dataset_path)

    def run_automl_workflow(self, file_path):
        """
        Main workflow for the AutoML application.
        """
        logging.info(f"Starting AutoML workflow for dataset: {file_path}")
        try:
            self.progress_updated.emit(10, "Reading dataset...")
            # Dataset loading logic: replace local CSV with public mirrors/fetchers
            # Example: For Iris
            # df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
            # For Titanic
            # df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
            # For Adult
            # df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None)
            # For MNIST
            # from sklearn.datasets import fetch_openml
            # mnist = fetch_openml('mnist_784', version=1, as_frame=True)
            # df = mnist.data
            # y = mnist.target
            # For Boston
            # from sklearn.datasets import fetch_openml
            # boston = fetch_openml(data_id=531, as_frame=True)
            # df = boston.data
            # y = boston.target
            # For California
            # from sklearn.datasets import fetch_california_housing
            # california = fetch_california_housing(as_frame=True)
            # df = california.data
            # y = california.target
            # For Bike Sharing
            # df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00275/day.csv")
            # For 20 Newsgroups
            # from sklearn.datasets import fetch_20newsgroups
            # newsgroups = fetch_20newsgroups(subset='train')
            # X = newsgroups.data
            # y = newsgroups.target
            # For SMS Spam
            # import zipfile, io, requests
            # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
            # with zipfile.ZipFile(io.BytesIO(requests.get(url).content)) as z:
            #     with z.open("SMSSpamCollection") as f:
            #         df = pd.read_csv(f, sep='\t', header=None, names=['label', 'message'])
            # For Heart Disease
            # df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header=None)
            # For Breast Cancer
            # from sklearn.datasets import load_breast_cancer
            # cancer = load_breast_cancer(as_frame=True)
            # df = cancer.data
            # y = cancer.target
            # For Wine Quality
            # df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
            # For Credit Card Fraud
            # df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")
            # Replace the following line with the appropriate dataset loader above for your use case:
            df = pd.read_csv(file_path)
            
            # 1. Data Analysis
            self.progress_updated.emit(25, "Analyzing data...")
            analyzer = DataAnalyzer(target_candidates=['target', 'label', 'y', 'class', 'output'])
            analysis_summary = analyzer.analyze_dataset(df)
            logging.info(f"Data analysis summary: {analysis_summary}")

            # 2. Meta-Learning
            self.progress_updated.emit(50, "Generating recommendations...")
            recommendations = self.meta_learner.recommend_algorithms(analysis_summary)
            logging.info(f"Meta-learning recommendations: {recommendations}")

            # 3. Pipeline Building
            self.progress_updated.emit(75, "Building pipeline...")
            builder = PipelineBuilder(recommendations, self.model_manager)
            
            if recommendations:
                best_algo_rec = recommendations[0]
                best_algo_name = best_algo_rec['model']
                
                target_column = analysis_summary.get('target_column')
                if not target_column:
                    raise ValueError("Target column could not be identified.")

                # Exclude target column from features
                features = df.drop(columns=[target_column])
                
                numerical_cols = features.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()

                preprocessor_steps = []
                if numerical_cols:
                    num_pipeline = Pipeline(steps=[
                        ('imputer', handle_missing(features[numerical_cols])),
                        ('scaler', scale_numerical(features[numerical_cols]))
                    ])
                    preprocessor_steps.append(('numerical', num_pipeline, numerical_cols))

                if categorical_cols:
                    cat_pipeline = Pipeline(steps=[
                        ('imputer', handle_missing(features[categorical_cols], strategy='most_frequent')),
                        ('encoder', encode_categorical(features[categorical_cols]))
                    ])
                    preprocessor_steps.append(('categorical', cat_pipeline, categorical_cols))

                preprocessor = ColumnTransformer(transformers=preprocessor_steps)

                pipeline = builder.create_pipeline(best_algo_name, [('preprocessor', preprocessor)])
                logging.info(f"Created pipeline: {pipeline}")

                # 4. Basic Training
                self.progress_updated.emit(90, "Training model...")
                X = features
                y = df[target_column]
                
                # This is a simplified training step. In a real scenario, you'd use train_test_split.
                pipeline.fit(X, y)
                logging.info("Pipeline training complete.")
                self.progress_updated.emit(100, "Workflow complete.")

            else:
                logging.warning("No recommendations from meta-learner.")
                self.progress_updated.emit(100, "Workflow complete - no recommendations.")

        except Exception as e:
            logging.error(f"An error occurred during the AutoML workflow: {e}", exc_info=True)
            self.progress_updated.emit(100, f"Error: {e}")


if __name__ == "__main__":
    # Set up logging first
    setup_logging()

    # Create and run the application
    app = QApplication(sys.argv)
    main_window = AutoMLApp()
    main_window.show()
    sys.exit(app.exec())