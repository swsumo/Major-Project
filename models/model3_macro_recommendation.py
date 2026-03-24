"""
MODEL 3: Macro Recommendation using Federated Learning
Recommends optimal protein intake based on user profile and goals
(Can be extended to carbs/fats later)
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

def prepare_macro_data(df):
    """
    Prepare data for macro recommendation
    Predict optimal protein target based on user profile and success
    """
    
    features = []
    targets = []
    
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('week')
        
        # Only use weeks where user had good adherence (successful weeks)
        # This teaches model what protein levels work for successful users
        successful_weeks = user_data[user_data['adherence_score'] > 0.7]
        
        if len(successful_weeks) < 2:
            continue
        
        for _, row in successful_weeks.iterrows():
            # Features: User profile
            feature_row = [
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
            ]
            
            # Target: Protein that worked well for this user
            # (avg_daily_protein from successful weeks)
            target = row['avg_daily_protein']
            
            features.append(feature_row)
            targets.append(target)
    
    return np.array(features), np.array(targets)

print("🔧 Preparing macro recommendation data...")
X, y = prepare_macro_data(df)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Protein range in data: {y.min():.1f}g - {y.max():.1f}g")

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
    print(f"   Client {i+1}: {len(X_client)} samples (Protein avg: {y_client.mean():.1f}g)")

# Train client models
print("\n🤖 Training FL Models (3 clients)...")

client_models = []

for i, (X_client, y_client) in enumerate(client_datasets):
    print(f"\n   Training Client {i+1}...")
    
    model = RandomForestRegressor(
        n_estimators=50, 
        max_depth=10,
        min_samples_split=5,
        random_state=42+i
    )
    model.fit(X_client, y_client)
    
    # Evaluate on client's data
    pred_train = model.predict(X_client)
    mae_train = mean_absolute_error(y_client, pred_train)
    
    print(f"   Client {i+1} MAE: {mae_train:.2f}g protein")
    
    client_models.append(model)

# Create ensemble model (Federated Averaging simulation)
print("\n🔄 Aggregating models (Federated Averaging)...")

class EnsembleMacroRecommender:
    """Ensemble of client models (simulates FL aggregation)"""
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        # Average predictions from all client models
        predictions = np.array([model.predict(X) for model in self.models])
        avg_pred = np.mean(predictions, axis=0)
        # Ensure reasonable protein range (40g - 250g)
        return np.clip(avg_pred, 40, 250)
    
    def recommend_macros(self, user_profile):
        """
        High-level function to recommend full macro split
        Input: user profile dict
        Output: {protein_g, carbs_g, fat_g, calories}
        """
        # This will be used by the agent
        protein_g = self.predict(user_profile.reshape(1, -1))[0]
        
        # Simple macro calculation (can be made more sophisticated)
        # Protein: 4 cal/g, Carbs: 4 cal/g, Fat: 9 cal/g
        
        # Assuming user_profile[12] is calorie_target
        calorie_target = user_profile[12]
        
        protein_calories = protein_g * 4
        
        # Fat: 25-30% of calories
        fat_calories = calorie_target * 0.27
        fat_g = fat_calories / 9
        
        # Carbs: remaining calories
        carb_calories = calorie_target - protein_calories - fat_calories
        carb_g = carb_calories / 4
        
        return {
            'protein_g': round(protein_g, 1),
            'carbs_g': round(carb_g, 1),
            'fat_g': round(fat_g, 1),
            'calories': round(calorie_target, 0)
        }

global_model = EnsembleMacroRecommender(client_models)

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

print(f"\nTraining MAE: {mae_train:.2f}g")
print(f"Test MAE: {mae_test:.2f}g")
print(f"Training RMSE: {rmse_train:.2f}g")
print(f"Test RMSE: {rmse_test:.2f}g")
print(f"Training R²: {r2_train:.3f}")
print(f"Test R²: {r2_test:.3f}")

# Save the global model
print("\n💾 Saving model...")
with open('models/trained_models/macro_recommendation_model.pkl', 'wb') as f:
    pickle.dump(global_model, f)

print("Model saved to 'models/trained_models/macro_recommendation_model.pkl'")

# Show some example predictions
print("\n🔍 Sample Protein Recommendations:")
print("="*80)
print(f"{'Actual Protein':<20} {'Recommended':<20} {'Error':<15} {'Interpretation'}")
print("="*80)

for i in range(10):
    actual = y_test[i]
    predicted = y_pred_test[i]
    error = abs(actual - predicted)
    
    # Interpret protein level
    if predicted > 150:
        interpretation = "High (Muscle gain/athletic)"
    elif predicted > 100:
        interpretation = "Moderate-High (Active)"
    elif predicted > 80:
        interpretation = "Moderate (Maintenance)"
    else:
        interpretation = "Light (Sedentary)"
    
    print(f"{actual:.1f}g{'':<13} {predicted:.1f}g{'':<13} {error:.2f}g{'':<10} {interpretation}")

print("="*80)

# Demonstrate full macro recommendation
print("\n🍽️  Example Full Macro Recommendations:")


# Example 1: Weight loss user
example_1 = np.array([30, 1, 85, 85, 1, 0, 0, 3, 0, 1, 0, 0, 1800])  # Male, 30yo, 85kg, weight loss
macros_1 = global_model.recommend_macros(example_1)
print(f"User: 30yo Male, 85kg, Weight Loss, 1800 cal")
print(f"  → Protein: {macros_1['protein_g']}g | Carbs: {macros_1['carbs_g']}g | Fat: {macros_1['fat_g']}g")

# Example 2: Muscle gain user
example_2 = np.array([25, 1, 70, 70, 0, 1, 0, 5, 0, 0, 0, 1, 2500])  # Male, 25yo, 70kg, muscle gain
macros_2 = global_model.recommend_macros(example_2)
print(f"\nUser: 25yo Male, 70kg, Muscle Gain, 2500 cal")
print(f"  → Protein: {macros_2['protein_g']}g | Carbs: {macros_2['carbs_g']}g | Fat: {macros_2['fat_g']}g")

# Example 3: Female maintenance
example_3 = np.array([28, 0, 60, 60, 0, 0, 1, 2, 0, 1, 0, 0, 1600])  # Female, 28yo, 60kg, maintenance
macros_3 = global_model.recommend_macros(example_3)
print(f"\nUser: 28yo Female, 60kg, Maintenance, 1600 cal")
print(f"  → Protein: {macros_3['protein_g']}g | Carbs: {macros_3['carbs_g']}g | Fat: {macros_3['fat_g']}g")



# Model interpretation
print("\n📈 Model Insights:")

print("This model helps the agent recommend:")
print("  • Optimal protein based on what worked for similar successful users")
print("  • Adjusts for goal (muscle gain needs more, maintenance needs less)")
print("  • Considers activity level and current weight")
print("  • Provides complete macro split (protein/carbs/fat) for meal planning")



