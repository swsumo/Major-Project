"""
Finalize Adherence Model (Binary Classification)
Run AFTER fl_adherence_server.py completes.

Usage: python fl_adherence_finalize.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report
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
            targets.append(1 if row['adherence_score'] >= 0.6 else 0)
    return np.array(features), np.array(targets)


class AdherenceEnsemble:
    """
    Weighted ensemble of 3 FL client classifiers.
    Binary: 0=Low adherence, 1=High adherence
    Returns probability (0-1) for agent compatibility.
    """
    def __init__(self, models, weights):
        self.models  = models
        self.weights = weights

    def predict(self, X):
        """Returns probability of high adherence (0-1) for agent."""
        probs = np.array([m.predict_proba(X)[:, 1] for m in self.models])
        return np.average(probs, axis=0, weights=self.weights)

    def predict_class(self, X):
        probs = self.predict(X)
        return (probs >= 0.5).astype(int)


def main():
    print("=" * 60)
    print("Finalizing Adherence Model (Binary Classification)")
    print("=" * 60)

    df = pd.read_csv('fl_training_data.csv')
    X, y = prepare_adherence_data(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Test set: Low={( y_test==0).sum()} High={(y_test==1).sum()}")

    # Load 3 client models
    client_models = []
    for i in range(1, 4):
        path = f'models/trained_models/adherence_client_{i}.pkl'
        if not os.path.exists(path):
            raise FileNotFoundError(f"Client {i} not found!")
        with open(path, 'rb') as f:
            client_models.append(pickle.load(f))
        print(f"Loaded Client {i}")

    # Evaluate each client
    print("\nIndividual client accuracies:")
    accs = []
    for i, model in enumerate(client_models):
        acc = accuracy_score(y_test, model.predict(X_test)) * 100
        accs.append(acc)
        print(f"   Client {i+1}: {acc:.1f}%")

    # Weighted ensemble
    weights = [a / sum(accs) for a in accs]
    print(f"Weights: {[round(w,3) for w in weights]}")

    ensemble = AdherenceEnsemble(client_models, weights)
    class_preds = ensemble.predict_class(X_test)
    acc = accuracy_score(y_test, class_preds) * 100

    print(f"\nFinal Ensemble Accuracy: {acc:.1f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, class_preds,
          target_names=['Low(<0.6)', 'High(>=0.6)']))

    # Save
    with open('models/trained_models/adherence_prediction_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    print("adherence_prediction_model.pkl saved!")

    # Cleanup
    for i in range(1, 4):
        path = f'models/trained_models/adherence_client_{i}.pkl'
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists('models/trained_models/adherence_fl_params.pkl'):
        os.remove('models/trained_models/adherence_fl_params.pkl')
    print("Cleaned up!")
    print(f"\nAdherence model ready! Accuracy: {acc:.1f}%")


if __name__ == "__main__":
    main()