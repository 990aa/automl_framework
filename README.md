# AutoML Framework

This project is a desktop-based Automated Machine Learning (AutoML) framework built with Python. It provides a user-friendly graphical interface for users to load datasets, receive intelligent recommendations for machine learning pipelines, and execute the training and evaluation process in parallel, leveraging the Dask library.

## Purpose

The main purpose of this framework is to simplify and accelerate the process of building and evaluating machine learning models. It aims to automate the repetitive and time-consuming tasks of data preprocessing, model selection, and hyperparameter tuning, making it easier for users to discover the optimal pipeline for their specific dataset. The inclusion of meta-learning allows the framework to learn from past experiments and improve its recommendations over time.

## Result

The application provides the following results:
- A comprehensive analysis of the input dataset.
- A set of recommended machine learning pipelines tailored to the dataset.
- A parallelized training process that tests multiple pipelines.
- A comparison view to analyze the performance metrics (e.g., accuracy, F1-score) of the trained models.
- The ability to save and load entire experiment sessions, including datasets and results.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/990aa/automl_framework.git
    cd automl_framework
    ```

2.  **Create and activate a virtual environment:**

    - **On Windows:**
      ```bash
      python -m venv .venv
      .venv\Scripts\activate
      ```

    - **On macOS/Linux:**
      ```bash
      python3 -m venv .venv
      source .venv/bin/activate
      ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

Once the setup is complete, you can run the application with the following command:

```bash
python main_app.py
```

This will launch the AutoML Framework's graphical user interface.
