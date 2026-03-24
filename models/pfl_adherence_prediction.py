"""
PERSONALIZED FEDERATED LEARNING - ADHERENCE PREDICTION MODEL
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

os.makedirs('models/trained_models/pfl', exist_ok=True)
df = pd.read_csv('fl_training_data.csv')

def prepare_adherence_data(df):
    features = []
    targets = []
    user_ids = []
    
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('week')
        for i in range(len(user_data)):
            row = user_data.iloc[i]
            calorie_deficit_pct = (row['tdee'] - row['calorie_target']) / row['tdee']
            feature_row = [
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
            ]
            features.append(feature_row)
            targets.append(row['adherence_score'])
            user_ids.append(user_id)
    
    return np.array(features), np.array(targets), user_ids

print("\n Preparing data...")
X, y, user_ids = prepare_adherence_data(df)

X_train, X_test, y_train, y_test, users_train, users_test = train_test_split(
    X, y, user_ids, test_size=0.2, random_state=42
)
print(f"Training: {len(X_train)}, Test: {len(X_test)}")

def split_for_fl_clients(X, y, users, n_clients=10):
    unique_users = list(set(users))
    np.random.shuffle(unique_users)
    users_per_client = len(unique_users) // n_clients
    client_data = []
    
    for i in range(n_clients):
        start = i * users_per_client
        end = None if i == n_clients - 1 else start + users_per_client
        client_users = unique_users[start:end]
        mask = np.isin(users, client_users)
        client_data.append((X[mask], y[mask]))
    
    return client_data

client_datasets = split_for_fl_clients(X_train, y_train, users_train, n_clients=10)

print("\nTraining 10 FL clients...")
client_models = []

for i, (X_c, y_c) in enumerate(client_datasets):
    model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42+i)
    model.fit(X_c, y_c)
    mae = mean_absolute_error(y_c, model.predict(X_c))
    print(f"   Client {i+1}: {len(X_c)} samples, MAE: {mae:.3f}")
    client_models.append(model)

class EnsembleGlobalModel:
    def __init__(self, models):
        self.models = models
    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return np.clip(np.mean(preds, axis=0), 0.0, 1.0)

global_model = EnsembleGlobalModel(client_models)

y_pred_global = global_model.predict(X_test)
mae_global = mean_absolute_error(y_test, y_pred_global)
r2_global = r2_score(y_test, y_pred_global)

print(f"\nGlobal Model: MAE={mae_global:.3f}, R²={r2_global:.3f}")

def create_personalized_models(df, global_model):
    personalized_models = {}
    
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('week')
        
        X_user = []
        y_user = []
        
        for i in range(len(user_data)):
            row = user_data.iloc[i]
            calorie_deficit_pct = (row['tdee'] - row['calorie_target']) / row['tdee']
            
            feature_row = [
                row['age'], 1 if row['gender'] == 'M' else 0, row['start_weight'],
                row['gym_days_target'], calorie_deficit_pct,
                1 if row['goal'] == 'weight_loss' else 0,
                1 if row['goal'] == 'muscle_gain' else 0,
                1 if row['activity_level'] == 'sedentary' else 0,
                1 if row['activity_level'] == 'active' else 0, row['week']
            ]
            
            X_user.append(feature_row)
            y_user.append(row['adherence_score'])
        
        if len(X_user) >= 3:
            personal_model = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42)
            personal_model.fit(np.array(X_user), np.array(y_user))
            personalized_models[user_id] = personal_model
    
    return personalized_models

personalized_models = create_personalized_models(df, global_model)
print(f"Created {len(personalized_models)} personalized models")

# PHASE 3: Hybrid pFL Model
class PersonalizedFLModel:
    def __init__(self, global_model, personalized_models, alpha=0.7):
        self.global_model = global_model
        self.personalized_models = personalized_models
        self.alpha = alpha
    
    def predict(self, X, user_id):
        global_pred = self.global_model.predict(X)
        if user_id in self.personalized_models:
            personal_pred = self.personalized_models[user_id].predict(X)
            return self.alpha * personal_pred + (1 - self.alpha) * global_pred
        return global_pred
    
    def predict_batch(self, X, user_ids):
        return np.array([self.predict(X[i:i+1], uid)[0] for i, uid in enumerate(user_ids)])

pfl_model = PersonalizedFLModel(global_model, personalized_models, alpha=0.7)

y_pred_pfl = pfl_model.predict_batch(X_test, users_test)
mae_pfl = mean_absolute_error(y_test, y_pred_pfl)
r2_pfl = r2_score(y_test, y_pred_pfl)

print(f"\nPersonalized FL: MAE={mae_pfl:.3f}, R²={r2_pfl:.3f}")

# Comparison

print("COMPARISON")

print(f"Global FL: MAE={mae_global:.3f}, R²={r2_global:.3f}")
print(f"Personalized FL: MAE={mae_pfl:.3f}, R²={r2_pfl:.3f}")
improvement = ((mae_global - mae_pfl) / mae_global) * 100
print(f"Improvement: {improvement:.1f}%")

# Save
with open('models/trained_models/pfl/global_adherence_model.pkl', 'wb') as f:
    pickle.dump(global_model, f)
with open('models/trained_models/pfl/personalized_adherence_models.pkl', 'wb') as f:
    pickle.dump(personalized_models, f)
with open('models/trained_models/pfl/pfl_adherence_model.pkl', 'wb') as f:
    pickle.dump(pfl_model, f)

print("\n Models saved!")

