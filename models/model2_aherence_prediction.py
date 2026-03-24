"""
MODEL 2: Adherence Prediction using Federated Learning
Predicts how likely a user is to stick to their fitness plan (0.0 to 1.0)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Create folders if they don't exist
os.makedirs('models/trained_models', exist_ok=True)

# Load data
print("📊 Loading training data...")
df = pd.read_csv('fl_training_data.csv')

def prepare_adherence_data(df):
    """
    Prepare data for adherence prediction
    Predict adherence based on user profile and plan difficulty
    """
    
    features = []
    targets = []
    
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('week')
        
        # Use first 2 weeks to establish baseline adherence
        if len(user_data) < 3:
            continue
        
        # For each week, predict adherence based on profile + current state
        for i in range(len(user_data)):
            row = user_data.iloc[i]
            
            # Calculate plan difficulty
            calorie_deficit_pct = (row['tdee'] - row['calorie_target']) / row['tdee']
            
            # Features
            feature_row = [
                row['age'],
                1 if row['gender'] == 'M' else 0,
                row['start_weight'],
                row['gym_days_target'],
                calorie_deficit_pct,  # How aggressive is the plan
                1 if row['goal'] == 'weight_loss' else 0,
                1 if row['goal'] == 'muscle_gain' else 0,
                1 if row['activity_level'] == 'sedentary' else 0,
                1 if row['activity_level'] == 'active' else 0,
                row['week']  # Week number (early weeks = higher adherence)
            ]
            
            # Target: actual adherence score
            target = row['adherence_score']
            
            features.append(feature_row)
            targets.append(target)
    
    return np.array(features), np.array(targets)

print("🔧 Preparing adherence prediction data...")
X, y = prepare_adherence_data(df)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Split training data into 3 FL clients
def split_for_fl_clients(X, y, n_clients=3):
    """Split data into chunks for FL clients"""
    client_data = []
    chunk_size = len(X) // n_clients
    
    for i in range(n_clients):
        start_idx = i * chunk_size
        if i == n_clients - 1:
            end_idx = len(X)
        else:
            end_idx = (i + 1) * chunk_size
        
        client_data.append((X[start_idx:end_idx], y[start_idx:end_idx]))
    
    return client_data

client_datasets = split_for_fl_clients(X_train, y_train, n_clients=3)

print(f"\n📦 Split data into 3 FL clients:")
for i, (X_client, y_client) in enumerate(client_datasets):
    print(f"   Client {i+1}: {len(X_client)} samples")

# Train client models
print("\n🤖 Training FL Models (3 clients)...")

client_models = []

for i, (X_client, y_client) in enumerate(client_datasets):
    print(f"\n   Training Client {i+1}...")
    
    model = RandomForestRegressor(
        n_estimators=50, 
        max_depth=8, 
        min_samples_split=10,
        random_state=42+i
    )
    model.fit(X_client, y_client)
    
    # Evaluate on client's data
    pred_train = model.predict(X_client)
    mae_train = mean_absolute_error(y_client, pred_train)
    
    print(f"   Client {i+1} MAE: {mae_train:.3f}")
    
    client_models.append(model)

# Create ensemble model (Federated Averaging simulation)
print("\n🔄 Aggregating models (Federated Averaging)...")

class EnsembleAdherencePredictor:
    """Ensemble of client models (simulates FL aggregation)"""
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        # Average predictions from all client models
        predictions = np.array([model.predict(X) for model in self.models])
        avg_pred = np.mean(predictions, axis=0)
        # Clip to valid adherence range [0, 1]
        return np.clip(avg_pred, 0.0, 1.0)

global_model = EnsembleAdherencePredictor(client_models)

# Test the global model
print("\n📊 Evaluating Global FL Model...")

y_pred_train = global_model.predict(X_train)
y_pred_test = global_model.predict(X_test)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"\nTraining MAE: {mae_train:.3f}")
print(f"Test MAE: {mae_test:.3f}")
print(f"Training RMSE: {rmse_train:.3f}")
print(f"Test RMSE: {rmse_test:.3f}")
print(f"Training R²: {r2_train:.3f}")
print(f"Test R²: {r2_test:.3f}")

# Save the global model
print("\n💾 Saving model...")
with open('models/trained_models/adherence_prediction_model.pkl', 'wb') as f:
    pickle.dump(global_model, f)

print("Model saved to 'models/trained_models/adherence_prediction_model.pkl'")

# Show some example predictions
print("\n🔍 Sample Predictions:")

print(f"{'Actual Adherence':<20} {'Predicted':<20} {'Error':<15} {'Status'}")


for i in range(10):
    actual = y_test[i]
    predicted = y_pred_test[i]
    error = abs(actual - predicted)
    
    # Interpret adherence
    if predicted > 0.75:
        status = "High (Likely to follow)"
    elif predicted > 0.5:
        status = "Medium (May struggle)"
    else:
        status = "Low (Likely to slack)"
    
    print(f"{actual:.2f} ({actual*100:.0f}%){'':<8} {predicted:.2f} ({predicted*100:.0f}%){'':<8} {error:.3f}{'':<10} {status}")



# Model interpretation
print("\n📈 Model Insights:")

print("This model helps the agent decide:")
print("  • If adherence < 0.6: Use easier plan (moderate calorie deficit)")
print("  • If adherence > 0.75: User can handle aggressive plan")
print("  • Helps prevent plan failure by matching difficulty to user capability")

print("MODEL 2 TRAINING COMPLETE! ✅")


