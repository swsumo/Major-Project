"""
Finalize Adherence Model after Flower FL training.
Run AFTER fl_adherence_server.py completes.

Usage: python fl_adherence_finalize.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def prepare_adherence_data(df):
    features, targets = [], []
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('week')
        if len(user_data) < 3:
            continue
        for _, row in user_data.iterrows():
            calorie_deficit_pct = (row['tdee'] - row['calorie_target']) / row['tdee']
            features.append([
                row['age'],
                1 if row['gender'] == 'M' else 0,
                row['start_weight'],
                row['gym_days_target'],
                calorie_deficit_pct,
                1 if row['goal'] == 'weight_loss' else 0,
                1 if row['goal'] == 'muscle_gain' else 0,
                1 if row['activity_level'] == 'sedentary' else 0,
                1 if row['activity_level'] == 'active' else 0,
                row['week']
            ])
            targets.append(row['adherence_score'])
    return np.array(features), np.array(targets)


class EnsembleAdherencePredictor:
    """Single best client model wrapped for compatibility with agent."""
    def __init__(self, model):
        self.models = [model]

    def predict(self, X):
        return np.clip(self.models[0].predict(X), 0.0, 1.0)


def main():
    print("=" * 60)
    print("Finalizing Adherence Model")
    print("=" * 60)

    df = pd.read_csv('fl_training_data.csv')
    X, y = prepare_adherence_data(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load 3 client models
    client_models = []
    for i in range(1, 4):
        path = f'models/trained_models/adherence_client_{i}.pkl'
        if not os.path.exists(path):
            raise FileNotFoundError(f"Client {i} model not found! Run FL training first.")
        with open(path, 'rb') as f:
            client_models.append(pickle.load(f))
        print(f"Loaded Adherence Client {i} model")

    # Find best individual client model
    print("\nEvaluating each client model individually:")
    best_model  = None
    best_mae    = float('inf')
    best_client = 0

    for i, model in enumerate(client_models):
        preds = np.clip(model.predict(X_test), 0, 1)
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        acc   = np.mean(np.abs(preds - y_test) < 0.1) * 100
        print(f"   Client {i+1}: MAE={mae:.4f} | R2={r2:.4f} | Accuracy={acc:.1f}%")
        if mae < best_mae:
            best_mae    = mae
            best_model  = model
            best_client = i + 1

    print(f"\nBest model: Client {best_client} (MAE: {best_mae:.4f})")

    # Wrap best model
    ensemble = EnsembleAdherencePredictor(best_model)
    preds    = ensemble.predict(X_test)
    mae      = mean_absolute_error(y_test, preds)
    r2       = r2_score(y_test, preds)
    accuracy = np.mean(np.abs(preds - y_test) < 0.1) * 100

    print(f"\nFinal Adherence Model Results:")
    print(f"   MAE:      {mae:.4f}")
    print(f"   R2:       {r2:.4f}")
    print(f"   Accuracy: {accuracy:.1f}%")

    # Save
    with open('models/trained_models/adherence_prediction_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    print("\nadherence_prediction_model.pkl saved!")

    # Cleanup
    for i in range(1, 4):
        path = f'models/trained_models/adherence_client_{i}.pkl'
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists('models/trained_models/adherence_fl_params.pkl'):
        os.remove('models/trained_models/adherence_fl_params.pkl')
    print("Cleaned up intermediate files")

    print(f"\nAdherence model ready!")
    print(f"   MAE: {mae:.4f} | Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()