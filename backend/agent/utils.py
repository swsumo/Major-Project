"""
AGENT UTILS: Helper classes and functions
Contains Ensemble model classes needed for unpickling
"""

import numpy as np
import pickle

# =============================================================================
# ENSEMBLE MODEL CLASSES (needed to load FL models)
# =============================================================================

class EnsembleWeightPredictor:
    """Ensemble of client models (simulates FL aggregation)"""
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)


class EnsembleAdherencePredictor:
    """Ensemble of client models (simulates FL aggregation)"""
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        avg_pred = np.mean(predictions, axis=0)
        return np.clip(avg_pred, 0.0, 1.0)


class EnsembleMacroRecommender:
    """Ensemble of client models (simulates FL aggregation)"""
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        avg_pred = np.mean(predictions, axis=0)
        return np.clip(avg_pred, 40, 250)
    
    def recommend_macros(self, user_profile):
        """
        High-level function to recommend full macro split
        Input: user profile array
        Output: {protein_g, carbs_g, fat_g, calories}
        """
        protein_g = self.predict(user_profile.reshape(1, -1))[0]
        calorie_target = user_profile[12]
        protein_calories = protein_g * 4
        fat_calories = calorie_target * 0.27
        fat_g = fat_calories / 9
        carb_calories = calorie_target - protein_calories - fat_calories
        carb_g = carb_calories / 4

        return {
            'protein_g': round(protein_g, 1),
            'carbs_g': round(carb_g, 1),
            'fat_g': round(fat_g, 1),
            'calories': round(calorie_target, 0)
        }


# =============================================================================
# MODEL LOADING HELPER
# =============================================================================

def load_fl_model(model_path: str):
    """
    Load a FL model with proper class definitions injected into __main__
    so pickle can find the classes during deserialization.
    """
    import __main__
    __main__.EnsembleWeightPredictor   = EnsembleWeightPredictor
    __main__.EnsembleAdherencePredictor = EnsembleAdherencePredictor
    __main__.EnsembleMacroRecommender  = EnsembleMacroRecommender

    with open(model_path, 'rb') as f:
        return pickle.load(f)