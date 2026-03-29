"""
Finalize Macro Recommendation Model after Flower FL training.
Run AFTER fl_macro_server.py completes.

Usage: python fl_macro_finalize.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def prepare_macro_data(df):
    features, targets = [], []
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('week')
        successful = user_data[user_data['adherence_score'] > 0.7]
        if len(successful) < 2:
            continue
        for _, row in successful.iterrows():
            features.append([
                row['age'],
                1 if row['gender'] == 'M' else 0,
                row['weight'],
                row['start_weight'],
                1 if row['goal'] == 'weight_loss' else 0,
                1 if row['goal'] == 'muscle_gain' else 0,
                1 if row['goal'] == 'maintenance' else 0,
                row['gym_days'],
                1 if row['activity_level'] == 'sedentary' else 0,
                1 if row['activity_level'] == 'light' else 0,
                1 if row['activity_level'] == 'moderate' else 0,
                1 if row['activity_level'] == 'active' else 0,
                row['calorie_target']
            ])
            targets.append(row['protein_target'])
    return np.array(features), np.array(targets)


class EnsembleMacroRecommender:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return np.clip(np.mean(preds, axis=0), 40, 250)

    def recommend_macros(self, user_profile):
        protein_g      = self.predict(user_profile.reshape(1, -1))[0]
        calorie_target = user_profile[12]
        fat_calories   = calorie_target * 0.27
        fat_g          = fat_calories / 9
        carb_calories  = calorie_target - (protein_g * 4) - fat_calories
        carb_g         = carb_calories / 4
        return {
            'protein_g': round(protein_g, 1),
            'carbs_g':   round(carb_g, 1),
            'fat_g':     round(fat_g, 1),
            'calories':  round(calorie_target, 0)
        }


def main():
    print("=" * 60)
    print("🔧 Finalizing Macro Recommendation Model")
    print("=" * 60)

    df = pd.read_csv('fl_training_data.csv')
    X, y = prepare_macro_data(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load 3 client models
    client_models = []
    for i in range(1, 4):
        path = f'models/trained_models/macro_client_{i}.pkl'
        if not os.path.exists(path):
            raise FileNotFoundError(f"Client {i} model not found! Run FL training first.")
        with open(path, 'rb') as f:
            client_models.append(pickle.load(f))
        print(f"✅ Loaded Macro Client {i} model")

    # Create ensemble
    ensemble = EnsembleMacroRecommender(client_models)
    preds    = ensemble.predict(X_test)
    mae      = mean_absolute_error(y_test, preds)
    r2       = r2_score(y_test, preds)

    print(f"\n📊 Macro Model Results:")
    print(f"   MAE: {mae:.4f}g protein")
    print(f"   R²:  {r2:.4f}")

    # Save
    with open('models/trained_models/macro_recommendation_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    print("\n✅ macro_recommendation_model.pkl saved!")

    # Cleanup
    for i in range(1, 4):
        path = f'models/trained_models/macro_client_{i}.pkl'
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists('models/trained_models/macro_fl_params.pkl'):
        os.remove('models/trained_models/macro_fl_params.pkl')
    print("✅ Cleaned up intermediate files")

    print(f"\n🎉 Macro model ready! MAE: {mae:.4f}g protein | R²: {r2:.4f}")


if __name__ == "__main__":
    main()