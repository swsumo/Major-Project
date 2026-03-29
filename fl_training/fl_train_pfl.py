import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_client_models():
    models = []
    for i in range(1, 4):
        path = f'models/trained_models/3client_pfl/client_{i}_model.pkl'
        if not os.path.exists(path):
            raise FileNotFoundError(f"Client {i} model not found at {path}. Run fl_server.py + fl_client.py first!")
        with open(path, 'rb') as f:
            models.append(pickle.load(f))
        print(f"Loaded Client {i} model")
    return models

class EnsembleGlobalModel:
    """Averages predictions from all 3 client models — this IS the global FL model."""
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = np.array([m.predict(X) for m in self.models])
        return np.mean(predictions, axis=0)

class PersonalizedFLModel:
    """
    Combines global model + personal model per user.
    70% personal + 30% global for known users.
    Falls back to global for new users.
    """
    def __init__(self, global_model, personalized_models, alpha=0.7):
        self.global_model = global_model
        self.personalized_models = personalized_models
        self.alpha  = alpha

    def predict(self, X, user_id=None):
        global_pred = self.global_model.predict(X)
        if user_id and user_id in self.personalized_models:
            personal_pred = self.personalized_models[user_id].predict(X)
            return self.alpha * personal_pred + (1 - self.alpha) * global_pred
        return global_pred

    def predict_batch(self, X, user_ids):
        return np.array([
            self.predict(X[i:i+1], uid)[0]
            for i, uid in enumerate(user_ids)
        ])

def prepare_weight_data(df):
    features, targets, user_ids = [], [], []
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('week')
        for i in range(len(user_data) - 1):
            curr = user_data.iloc[i]
            nxt  = user_data.iloc[i + 1]
            features.append([
                curr['weight'], curr['avg_daily_calories'],
                curr['avg_daily_protein'], curr['gym_days'],
                curr['age'], 1 if curr['gender'] == 'M' else 0,
                curr['tdee']
            ])
            targets.append(nxt['weight'])
            user_ids.append(user_id)
    return np.array(features), np.array(targets), user_ids


def create_personalized_models(df, scaler):
    """Fine-tune a personal model for each user on their own data."""
    personalized_models = {}

    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('week')
        X_user, y_user = [], []

        for i in range(len(user_data) - 1):
            curr = user_data.iloc[i]
            nxt  = user_data.iloc[i + 1]
            X_user.append([
                curr['weight'], curr['avg_daily_calories'],
                curr['avg_daily_protein'], curr['gym_days'],
                curr['age'], 1 if curr['gender'] == 'M' else 0,
                curr['tdee']
            ])
            y_user.append(nxt['weight'])

        X_user = np.array(X_user)
        y_user = np.array(y_user)

        if len(X_user) < 3:
            continue

        X_user_scaled = scaler.transform(X_user)

        personal_model = RandomForestRegressor(
            n_estimators=50, max_depth=6,
            min_samples_split=3, min_samples_leaf=2,
            random_state=42
        )
        personal_model.fit(X_user_scaled, y_user)
        personalized_models[user_id] = personal_model
    return personalized_models

def main():
    print("POST-FEDERATION: Creating Personalized FL Model")
    # Load data
    df     = pd.read_csv('fl_training_data.csv')
    X, y, user_ids = prepare_weight_data(df)
    # Load scaler saved by Client 1
    with open('models/trained_models/3client_pfl/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print(" Scaler loaded")
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test, users_train, users_test = train_test_split(
        X_scaled, y, user_ids, test_size=0.2, random_state=42
    )
    # Load 3 client models from flwr training
    client_models = load_client_models()
    # Create ensemble global model
    global_model = EnsembleGlobalModel(client_models)
    global_preds = global_model.predict(X_test)
    global_mae   = mean_absolute_error(y_test, global_preds)
    global_r2    = r2_score(y_test, global_preds)
    print(f"\n Global FL Model (ensemble of 3 clients):")
    print(f"   MAE: {global_mae:.4f} kg | R²: {global_r2:.4f}")

    # Create personalized models
    print("\n Creating personalized models for each user...")
    personalized_models = create_personalized_models(df, scaler)
    print(f" Created {len(personalized_models)} personalized models")

    # Create final pFL model
    pfl_model  = PersonalizedFLModel(global_model, personalized_models, alpha=0.7)
    pfl_preds  = pfl_model.predict_batch(X_test, users_test)
    pfl_mae    = mean_absolute_error(y_test, pfl_preds)
    pfl_r2     = r2_score(y_test, pfl_preds)
    improvement = ((global_mae - pfl_mae) / global_mae) * 100
    print(f"\n Personalized FL Model:")
    print(f" MAE: {pfl_mae:.4f} kg | R²: {pfl_r2:.4f}")
    print(f" Improvement over global FL: {improvement:.1f}%")

    # ── Save final models ──────────────────────────────────────────────────────
    print("\n Saving final models...")
    with open('models/trained_models/3client_pfl/pfl_model.pkl', 'wb') as f:
        pickle.dump(pfl_model, f)
    print(" pfl_model.pkl saved")
    # Save metadata
    metadata = {
        'n_clients':        3,
        'fl_framework':     'Flower (flwr)',
        'fl_rounds':        5,
        'global_mae':       global_mae,
        'global_r2':        global_r2,
        'pfl_mae':          pfl_mae,
        'pfl_r2':           pfl_r2,
        'improvement_pct':  improvement,
        'n_personal_models': len(personalized_models),
    }
    with open('models/trained_models/3client_pfl/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print(" metadata.pkl saved")

    # Clean up intermediate client model files
    for i in range(1, 4):
        path = f'models/trained_models/3client_pfl/client_{i}_model.pkl'
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists('models/trained_models/3client_pfl/fl_global_params.pkl'):
        os.remove('models/trained_models/3client_pfl/fl_global_params.pkl')
    print("Cleaned up intermediate files")
    print(f"\nFinal Results:")
    print(f"   Global FL MAE:       {global_mae:.4f} kg")
    print(f"   Personalized FL MAE: {pfl_mae:.4f} kg")
    print(f"   Improvement:         {improvement:.1f}%")
    print(f"\n Saved: models/trained_models/3client_pfl/pfl_model.pkl")

if __name__ == "__main__":
    main() 