import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

os.makedirs('models/trained_models/3client_pfl', exist_ok=True)

# Load data
df = pd.read_csv('fl_training_data.csv')

def prepare_weight_data(df):
    features = []
    targets = []
    user_ids = []
    
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('week')
        
        for i in range(len(user_data) - 1):
            current_week = user_data.iloc[i]
            next_week = user_data.iloc[i + 1]
            
            feature_row = [
                current_week['weight'],
                current_week['avg_daily_calories'],
                current_week['avg_daily_protein'],
                current_week['gym_days'],
                current_week['age'],
                1 if current_week['gender'] == 'M' else 0,
                current_week['tdee']
            ]
            
            features.append(feature_row)
            targets.append(next_week['weight'])
            user_ids.append(user_id)
    
    return np.array(features), np.array(targets), user_ids

print("\n" + "="*80)
print("🔧 FEDERATED LEARNING WITH 3 CLIENTS (More Data Per Client)")
print("="*80)

X, y, user_ids = prepare_weight_data(df)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test, users_train, users_test = train_test_split(
    X_scaled, y, user_ids, test_size=0.2, random_state=42
)

print(f"\nData split:")
print(f"  Training: {len(X_train)} samples")
print(f"  Test: {len(X_test)} samples")
print(f"  Total users: {len(set(user_ids))}")

def split_for_fl_clients(X, y, users, n_clients=3):
    unique_users = list(set(users))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_users)
    users_per_client = len(unique_users) // n_clients
    client_data = []
    
    for i in range(n_clients):
        start = i * users_per_client
        end = None if i == n_clients - 1 else start + users_per_client
        client_users = unique_users[start:end]
        mask = np.isin(users, client_users)
        client_data.append((X[mask], y[mask], client_users))
    
    return client_data

client_datasets = split_for_fl_clients(X_train, y_train, users_train, n_clients=3)

print(f"\n✅ Split into 3 FL clients:")
for i, (X_client, y_client, client_users) in enumerate(client_datasets):
    print(f"   Client {i+1}: {len(X_client)} samples from {len(client_users)} users")

# ========================================
# FIRST: Test with Centralized Baseline
# ========================================
print("\n" + "="*80)
print("📊 BASELINE: Centralized Model (For Comparison)")
print("="*80)

centralized = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
centralized.fit(X_train, y_train)

y_pred_cent = centralized.predict(X_test)
mae_cent = mean_absolute_error(y_test, y_pred_cent)
r2_cent = r2_score(y_test, y_pred_cent)

print(f"\nCentralized Model:")
print(f"   Test MAE: {mae_cent:.3f} kg")
print(f"   Test R²: {r2_cent:.3f}")

# ========================================
# STRATEGY 1: Regularized Random Forest
# ========================================
print("\n" + "="*80)
print("STRATEGY 1: REGULARIZED RANDOM FOREST")
print("="*80)

client_models_rf = []

for i, (X_client, y_client, _) in enumerate(client_datasets):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42+i
    )
    model.fit(X_client, y_client)
    
    own_mae = mean_absolute_error(y_client, model.predict(X_client))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    
    print(f"   Client {i+1}: Train={own_mae:.3f} kg | Test={test_mae:.3f} kg")
    client_models_rf.append(model)

class EnsembleGlobalModel:
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)

global_rf = EnsembleGlobalModel(client_models_rf)
y_pred_rf = global_rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"\n📊 Global FL Model (RF):")
print(f"   Test MAE: {mae_rf:.3f} kg")
print(f"   Test R²: {r2_rf:.3f}")

# ========================================
# STRATEGY 2: Gradient Boosting
# ========================================
print("\n" + "="*80)
print("STRATEGY 2: GRADIENT BOOSTING")
print("="*80)

client_models_gb = []

for i, (X_client, y_client, _) in enumerate(client_datasets):
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42+i
    )
    model.fit(X_client, y_client)
    
    own_mae = mean_absolute_error(y_client, model.predict(X_client))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    
    print(f"   Client {i+1}: Train={own_mae:.3f} kg | Test={test_mae:.3f} kg")
    client_models_gb.append(model)

global_gb = EnsembleGlobalModel(client_models_gb)
y_pred_gb = global_gb.predict(X_test)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f"\n📊 Global FL Model (GB):")
print(f"   Test MAE: {mae_gb:.3f} kg")
print(f"   Test R²: {r2_gb:.3f}")

# ========================================
# Choose best global model
# ========================================
print("\n" + "="*80)
print("📊 GLOBAL MODEL COMPARISON")
print("="*80)

print(f"\n{'Model':<30} {'MAE (kg)':<12} {'R²':<10}")
print("-"*80)
print(f"{'Centralized (baseline)':<30} {mae_cent:<12.3f} {r2_cent:<10.3f}")
print(f"{'FL - Random Forest':<30} {mae_rf:<12.3f} {r2_rf:<10.3f}")
print(f"{'FL - Gradient Boosting':<30} {mae_gb:<12.3f} {r2_gb:<10.3f}")

# Pick best FL model
if mae_rf < mae_gb:
    best_global = global_rf
    best_mae_global = mae_rf
    best_r2_global = r2_rf
    best_name = "Random Forest"
    best_client_models = client_models_rf
else:
    best_global = global_gb
    best_mae_global = mae_gb
    best_r2_global = r2_gb
    best_name = "Gradient Boosting"
    best_client_models = client_models_gb

print(f"\n✅ BEST GLOBAL MODEL: FL - {best_name}")
print(f"   MAE: {best_mae_global:.3f} kg")
print(f"   R²: {best_r2_global:.3f}")

# ========================================
# Create Personalized Models
# ========================================
print("\n" + "="*80)
print("👤 CREATING PERSONALIZED MODELS")
print("="*80)

def create_personalized_models(df, scaler):
    personalized_models = {}
    
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('week')
        X_user = []
        y_user = []
        
        for i in range(len(user_data) - 1):
            current_week = user_data.iloc[i]
            next_week = user_data.iloc[i + 1]
            feature_row = [
                current_week['weight'],
                current_week['avg_daily_calories'],
                current_week['avg_daily_protein'],
                current_week['gym_days'],
                current_week['age'],
                1 if current_week['gender'] == 'M' else 0,
                current_week['tdee']
            ]
            X_user.append(feature_row)
            y_user.append(next_week['weight'])
        
        X_user = np.array(X_user)
        y_user = np.array(y_user)
        
        if len(X_user) < 3:
            continue
        
        X_user_scaled = scaler.transform(X_user)
        
        personal_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=6,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42
        )
        personal_model.fit(X_user_scaled, y_user)
        personalized_models[user_id] = personal_model
    
    return personalized_models

personalized_models = create_personalized_models(df, scaler)
print(f"✅ Created {len(personalized_models)} personalized models")

# ========================================
# Personalized FL Model
# ========================================
class PersonalizedFLModel:
    def __init__(self, global_model, personalized_models, alpha=0.7):
        self.global_model = global_model
        self.personalized_models = personalized_models
        self.alpha = alpha
    
    def predict(self, X, user_id):
        global_pred = self.global_model.predict(X)
        
        if user_id in self.personalized_models:
            personal_pred = self.personalized_models[user_id].predict(X)
            final_pred = self.alpha * personal_pred + (1 - self.alpha) * global_pred
        else:
            final_pred = global_pred
        
        return final_pred
    
    def predict_batch(self, X, user_ids):
        predictions = []
        for i, user_id in enumerate(user_ids):
            pred = self.predict(X[i:i+1], user_id)
            predictions.append(pred[0])
        return np.array(predictions)

pfl_model = PersonalizedFLModel(best_global, personalized_models, alpha=0.7)

print("\n📊 Evaluating Personalized FL Model...")
y_pred_pfl = pfl_model.predict_batch(X_test, users_test)
mae_pfl = mean_absolute_error(y_test, y_pred_pfl)
r2_pfl = r2_score(y_test, y_pred_pfl)

print(f"\n✅ PERSONALIZED FL (3 clients):")
print(f"   Test MAE: {mae_pfl:.3f} kg")
print(f"   Test R²: {r2_pfl:.3f}")

# ========================================
# Final Comparison
# ========================================
print("\n" + "="*80)
print("🏆 FINAL COMPARISON")
print("="*80)

print(f"\n{'Model Type':<35} {'MAE (kg)':<12} {'R²':<10}")
print("-"*80)
print(f"{'Centralized (baseline)':<35} {mae_cent:<12.3f} {r2_cent:<10.3f}")
print(f"{'FL Global (3 clients)':<35} {best_mae_global:<12.3f} {best_r2_global:<10.3f}")
print(f"{'Personalized FL (pFL)':<35} {mae_pfl:<12.3f} {r2_pfl:<10.3f}")

improvement = ((best_mae_global - mae_pfl) / best_mae_global) * 100
print(f"\n📈 Personalization Improvement: {improvement:.1f}%")

# ========================================
# Assessment
# ========================================
print("\n" + "="*80)
print("🎯 ASSESSMENT")
print("="*80)

if best_mae_global < 1.0:
    print("\n✅ EXCELLENT: Global model MAE < 1.0 kg")
    print("   → FL works well with 3 clients")
    print("   → Ready for DP implementation!")
elif best_mae_global < 2.0:
    print("\n✅ GOOD: Global model MAE < 2.0 kg")
    print("   → Significant improvement from 10 clients")
    print("   → Acceptable for research project")
    print("   → Can proceed to DP")
elif best_mae_global < 3.0:
    print("\n⚠️  MODERATE: Global model MAE < 3.0 kg")
    print("   → Better than 10 clients, but not ideal")
    print("   → Consider generating more data")
else:
    print("\n❌ POOR: Global model MAE > 3.0 kg")
    print("   → Need more data generation")

if mae_pfl < 0.5:
    print(f"\n🎉 OUTSTANDING: pFL MAE = {mae_pfl:.3f} kg (Target achieved!)")
elif mae_pfl < 1.0:
    print(f"\n✅ VERY GOOD: pFL MAE = {mae_pfl:.3f} kg")
elif mae_pfl < 2.0:
    print(f"\n✅ GOOD: pFL MAE = {mae_pfl:.3f} kg")
else:
    print(f"\n⚠️  pFL MAE = {mae_pfl:.3f} kg (Could be better)")

# ========================================
# Save Models
# ========================================
print("\n💾 Saving models...")

with open('models/trained_models/3client_pfl/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/trained_models/3client_pfl/global_model.pkl', 'wb') as f:
    pickle.dump(best_global, f)

with open('models/trained_models/3client_pfl/personalized_models.pkl', 'wb') as f:
    pickle.dump(personalized_models, f)

with open('models/trained_models/3client_pfl/pfl_model.pkl', 'wb') as f:
    pickle.dump(pfl_model, f)

metadata = {
    'n_clients': 3,
    'best_strategy': best_name,
    'centralized_mae': mae_cent,
    'centralized_r2': r2_cent,
    'global_mae': best_mae_global,
    'global_r2': best_r2_global,
    'pfl_mae': mae_pfl,
    'pfl_r2': r2_pfl,
}

with open('models/trained_models/3client_pfl/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("✅ Models saved!")

print("\n" + "="*80)
print("🎉 3-CLIENT FL TRAINING COMPLETE!")
print("="*80)

print(f"\n📊 Summary:")
print(f"   • 3 FL clients (~{len(X_train)//3} samples each)")
print(f"   • Global FL: {best_mae_global:.3f} kg MAE, R²={best_r2_global:.3f}")
print(f"   • Personalized FL: {mae_pfl:.3f} kg MAE, R²={r2_pfl:.3f}")
print(f"   • Improvement: {improvement:.1f}% from global to pFL")

print("\n📁 Saved to: models/trained_models/3client_pfl/")
print("\n" + "="*80)

