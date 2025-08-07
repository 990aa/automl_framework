import dask.dataframe as dd
import pandas as pd
import logging

class DataAnalyzer:
    """
    Analyzes a Dask DataFrame to extract metadata and identify key characteristics.
    """
    def __init__(self, target_candidates=None):
        self.target_candidates = target_candidates or ['target', 'class', 'label', 'y']

    def analyze_dataset(self, ddf):
        """
        Performs a full analysis of a Dask DataFrame.

        Args:
            ddf (dd.DataFrame): The Dask DataFrame to analyze.

        Returns:
            dict: A dictionary summarizing the dataset's properties.
        """
        logging.info("Analyzing Dask DataFrame...")
        
        # Persist the dataframe in memory for more efficient computation
        ddf = ddf.persist()

        n_rows = len(ddf)
        n_features = len(ddf.columns)
        
        # Compute column-wise stats
        missing_values = ddf.isnull().sum().compute()
        unique_values = ddf.nunique().compute()
        
        # Identify target column
        target_column, problem_type = self._identify_target_and_problem_type(ddf, unique_values)

        summary = {
            'n_rows': n_rows,
            'n_features': n_features - 1 if target_column else n_features,
            'target_column': target_column,
            'problem_type': problem_type,
            'missing_values': missing_values.to_dict(),
            'unique_values': unique_values.to_dict(),
            'columns': ddf.columns.tolist(),
            'dtypes': ddf.dtypes.to_dict()
        }
        
        logging.info(f"Analysis complete. Target: '{target_column}', Problem: '{problem_type}'")
        return summary

    def _identify_target_and_problem_type(self, ddf, unique_values):
        """
        Heuristically identifies the target column and problem type.
        """
        # 1. Check for explicit candidates
        for col_name in self.target_candidates:
            if col_name in ddf.columns:
                return self._classify_column(ddf[col_name], unique_values[col_name])

        # 2. Fallback: guess based on column properties (e.g., last column)
        last_col = ddf.columns[-1]
        return self._classify_column(ddf[last_col], unique_values[last_col])

    def _classify_column(self, column_series, n_unique):
        """
        Determines problem type based on a potential target column's properties.
        """
        col_name = column_series.name
        
        # Regression if numeric and has many unique values
        if pd.api.types.is_numeric_dtype(column_series.dtype) and n_unique > 50:
            return col_name, 'regression'
        
        # Classification if categorical or integer with few unique values
        if pd.api.types.is_categorical_dtype(column_series.dtype) or \
           (pd.api.types.is_integer_dtype(column_series.dtype) and n_unique <= 50):
            return col_name, 'classification'
            
        # Default fallback
        return col_name, 'unknown'