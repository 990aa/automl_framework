import json
import logging
import numpy as np

class MetaLearner:
    """
    Recommends algorithms by comparing the current dataset to past experiences.
    """
    def __init__(self, experiences_path='data/past_experiences.json'):
        self.experiences = self.load_experiences(experiences_path)
        logging.info(f"MetaLearner initialized with {len(self.experiences)} past experiences.")

    def load_experiences(self, filepath: str):
        """Loads the JSON database of past experiences."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Meta-learning experiences file not found at: {filepath}")
            return []
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from: {filepath}")
            return []

    def calculate_similarity(self, current_profile: dict, past_profile: dict):
        """
        Calculates a similarity score between the current dataset and a past one.
        The score is a weighted sum of differences, where a lower score means more similar.
        """
        score = 0
        weights = {
            'problem_type': 5,
            'n_features': 1,
            'n_rows': 0.5,
            'missing_ratio': 2,
            'feature_type_mismatch': 3
        }

        # 1. Problem Type (Critical)
        if current_profile.get('problem_type') != past_profile.get('problem_type'):
            return float('inf') # Not comparable if problem types differ

        # 2. Number of Features (Normalized Difference)
        score += weights['n_features'] * abs(current_profile['n_features'] - past_profile['n_features']) / max(current_profile['n_features'], past_profile['n_features'])

        # 3. Number of Rows (Normalized Difference, log scale)
        score += weights['n_rows'] * abs(np.log1p(current_profile['n_rows']) - np.log1p(past_profile['n_rows'])) / max(np.log1p(current_profile['n_rows']), np.log1p(past_profile['n_rows']))

        # 4. Missing Value Ratio
        current_missing = current_profile['missing_value_info']['overall_missing_ratio']
        past_missing = past_profile['missing_value_info']['overall_missing_ratio']
        score += weights['missing_ratio'] * abs(current_missing - past_missing)

        # 5. Feature Profile Mismatch
        current_fp = self._summarize_feature_profiles(current_profile['feature_profiles'])
        past_fp = past_profile['feature_profiles'] # Stored profile is already summarized
        
        mismatch = 0
        all_keys = set(current_fp.keys()) | set(past_fp.keys())
        for key in all_keys:
            mismatch += abs(current_fp.get(key, 0) - past_fp.get(key, 0))
        
        # Normalize mismatch by total number of features
        total_features = current_profile['n_features']
        if total_features > 0:
            score += weights['feature_type_mismatch'] * (mismatch / total_features)

        return score

    def _summarize_feature_profiles(self, detailed_profiles: dict):
        """Helper to count feature types from a detailed profile."""
        summary = {'numerical': 0, 'categorical': 0, 'text': 0, 'other': 0}
        for profile in detailed_profiles.values():
            summary[profile['type']] += 1
        return summary

    def find_most_similar(self, current_dataset_profile: dict):
        """
        Finds the most similar past experience and returns its recommendation.
        """
        if not self.experiences:
            return None, float('inf')

        best_match = None
        lowest_score = float('inf')

        for experience in self.experiences:
            past_dataset_profile = experience['dataset_profile']
            score = self.calculate_similarity(current_dataset_profile, past_dataset_profile)
            
            logging.info(f"Comparing to '{experience['project_name']}': Similarity score = {score:.4f}")

            if score < lowest_score:
                lowest_score = score
                best_match = experience
        
        return best_match, lowest_score

    def recommend_algorithms(self, current_dataset_profile: dict, top_n: int = 3):
        """
        Recommends a ranked list of algorithms based on the top N most similar experiences.
        """
        if not self.experiences:
            return []

        # Calculate similarity scores for all experiences
        scored_experiences = []
        for experience in self.experiences:
            # Ensure problem types match before calculating score
            if current_dataset_profile.get('problem_type') == experience.get('dataset_profile', {}).get('problem_type'):
                score = self.calculate_similarity(current_dataset_profile, experience['dataset_profile'])
                if score != float('inf'):
                    scored_experiences.append((score, experience))
        
        # Sort by score (lower is better)
        scored_experiences.sort(key=lambda x: x[0])

        # Aggregate recommendations from the top N
        recommendations = {}
        
        # Take top_n, or fewer if not enough experiences are available
        num_to_consider = min(top_n, len(scored_experiences))
        
        for score, experience in scored_experiences[:num_to_consider]:
            pipeline = experience.get('best_pipeline', {})
            model = pipeline.get('model')
            if not model:
                continue

            # Use a tuple of (model, preprocessing) as the key to group similar pipelines
            key = (model, tuple(sorted(pipeline.get('preprocessing', []))))
            
            if key not in recommendations:
                recommendations[key] = {
                    'model': model,
                    'preprocessing': pipeline.get('preprocessing', []),
                    'evidence_count': 0,
                    'source_projects': []
                }
            
            recommendations[key]['evidence_count'] += 1
            recommendations[key]['source_projects'].append(experience['project_name'])

        # Convert to a list and sort by evidence count (higher is better)
        ranked_list = sorted(recommendations.values(), key=lambda x: x['evidence_count'], reverse=True)
        
        logging.info(f"Generated {len(ranked_list)} recommendations based on top {num_to_consider} similar projects.")
        return ranked_list