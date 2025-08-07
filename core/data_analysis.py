import pandas as pd
import numpy as np
import logging

class DataAnalyzer:
    """
    Analyzes a dataset to extract metadata, feature profiles, and suggest a problem type.
    """
    def __init__(self, target_candidates=None):
        if target_candidates is None:
            self.target_candidates = ['target', 'label', 'y', 'class', 'output']
        else:
            self.target_candidates = target_candidates
        logging.info("DataAnalyzer initialized.")

    def analyze_dataset(self, df: pd.DataFrame):
        """
        Performs a comprehensive analysis of the given DataFrame.
        """
        logging.info(f"Starting dataset analysis for a dataframe with shape {df.shape}.")
        
        feature_profiles = self._create_feature_profiles(df)
        target_column, problem_type = self._infer_problem_type(df, feature_profiles)

        dataset_profile = {
            'n_rows': df.shape[0],
            'n_features': df.shape[1],
            'missing_value_info': self._analyze_missing_values(df),
            'feature_profiles': feature_profiles,
            'data_quality': self._assess_data_quality(df, target_column),
            'target_column': target_column,
            'problem_type': problem_type
        }
        
        logging.info(f"Dataset analysis complete. Inferred problem type: {problem_type}")
        return dataset_profile

    def _analyze_missing_values(self, df: pd.DataFrame):
        """Calculates missing value statistics."""
        total_missing = df.isnull().sum().sum()
        total_cells = np.prod(df.shape)
        overall_missing_ratio = total_missing / total_cells
        
        missing_per_column = {col: df[col].isnull().sum() for col in df.columns}
        missing_ratio_per_column = {col: val / df.shape[0] for col, val in missing_per_column.items()}

        return {
            'overall_missing_ratio': overall_missing_ratio,
            'missing_ratio_per_column': missing_ratio_per_column
        }

    def _create_feature_profiles(self, df: pd.DataFrame):
        """Creates a detailed profile for each feature."""
        profiles = {}
        for col in df.columns:
            dtype = df[col].dtype
            nunique = df[col].nunique()
            missing_ratio = df[col].isnull().sum() / len(df)
            
            profile = {'dtype': str(dtype), 'nunique': nunique, 'missing_ratio': missing_ratio}

            if pd.api.types.is_numeric_dtype(dtype):
                if nunique <= 2: # Binary is always categorical
                    profile['type'] = 'categorical'
                elif pd.api.types.is_integer_dtype(dtype) and nunique < 20: # Low-cardinality integer is categorical
                    profile['type'] = 'categorical'
                else:
                    profile['type'] = 'numerical'
                    profile['std'] = df[col].std()
                    profile['min'] = df[col].min()
                    profile['max'] = df[col].max()
            elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > 25 and nunique / len(df) > 0.8: # High unique, long strings
                     profile['type'] = 'text'
                else:
                     profile['type'] = 'categorical'
            else:
                profile['type'] = 'other'
            
            profiles[col] = profile
        return profiles

    def _infer_problem_type(self, df: pd.DataFrame, feature_profiles):
        """Infers the problem type (Classification/Regression) based on the target column."""
        target_column = None
        for col_name in self.target_candidates:
            if col_name in df.columns:
                target_column = col_name
                break
        
        if not target_column:
            # If no candidate found, assume unsupervised
            return None, 'Unsupervised'

        target_profile = feature_profiles[target_column]
        
        # If the identified target is numeric (but not low-cardinality int), it's regression.
        if target_profile['type'] == 'numerical':
            return target_column, 'Regression'
        
        # Otherwise, it's classification (categorical, binary, etc.)
        if target_profile['type'] == 'categorical':
            return target_column, 'Classification'
        
        return target_column, 'Unknown'

    def _assess_data_quality(self, df: pd.DataFrame, target_column: str = None):
        """Performs basic data quality checks."""
        quality_report = {}
        
        # Outlier detection using IQR (on non-target columns)
        outlier_info = {}
        numeric_cols = df.select_dtypes(include=np.number).columns
        if target_column and target_column in numeric_cols:
            numeric_cols = numeric_cols.drop(target_column)

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0: # Avoid division by zero or constant columns
                outlier_condition = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
                outliers = df[col][outlier_condition]
                if not outliers.empty:
                    outlier_info[col] = {'count': len(outliers), 'ratio': len(outliers) / len(df)}
        quality_report['outliers'] = outlier_info

        # Near-zero variance features
        near_zero_var = []
        for col in df.columns:
            if col == target_column: continue # Don't check target column
            
            nunique = df[col].nunique()
            if nunique <= 1:
                near_zero_var.append(col)
            elif nunique / len(df) < 0.01: # Less than 1% unique values
                near_zero_var.append(col)

        quality_report['near_zero_variance_features'] = near_zero_var
        
        return quality_report
