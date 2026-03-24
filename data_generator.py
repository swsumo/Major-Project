import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

def calculate_tdee(weight, height, age, gender, activity_level):
    """Calculate Total Daily Energy Expenditure using Mifflin-St Jeor"""
    if gender == 'M':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    activity_multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9
    }
    
    return bmr * activity_multipliers[activity_level]

def generate_users(n_users=90):
    """Generate 30 diverse user profiles"""
    users = []
    
    goals = ['weight_loss', 'muscle_gain', 'maintenance']
    activity_levels = ['sedentary', 'light', 'moderate', 'active']
    
    for i in range(1, n_users + 1):
        gender = np.random.choice(['M', 'F'])
        
        # Diverse age distribution
        age = np.random.randint(18, 55)
        
        # Realistic weight and height
        if gender == 'M':
            height = np.random.randint(165, 185)  # cm
            weight = np.random.randint(65, 105)   # kg
        else:
            height = np.random.randint(152, 172)
            weight = np.random.randint(50, 85)
        
        goal = np.random.choice(goals)
        activity_level = np.random.choice(activity_levels)
        
        # Gym days based on activity level
        gym_days_map = {
            'sedentary': np.random.randint(0, 2),
            'light': np.random.randint(2, 4),
            'moderate': np.random.randint(3, 5),
            'active': np.random.randint(4, 6)
        }
        gym_days = gym_days_map[activity_level]
        
        # Calculate TDEE
        tdee = calculate_tdee(weight, height, age, gender, activity_level)
        
        # Set calorie target based on goal
        if goal == 'weight_loss':
            calorie_target = tdee - np.random.randint(300, 600)
        elif goal == 'muscle_gain':
            calorie_target = tdee + np.random.randint(200, 400)
        else:
            calorie_target = tdee
        
        # Protein target (g per kg bodyweight)
        protein_multiplier = {
            'weight_loss': 1.8,
            'muscle_gain': 2.2,
            'maintenance': 1.4
        }
        protein_target = weight * protein_multiplier[goal]
        
        # Adherence personality (affects consistency)
        # High adherence = disciplined, Low = struggles with consistency
        base_adherence = np.random.uniform(0.5, 0.95)
        
        users.append({
            'user_id': f'user_{i:02d}',
            'age': age,
            'gender': gender,
            'height': height,
            'start_weight': weight,
            'goal': goal,
            'activity_level': activity_level,
            'gym_days_target': gym_days,
            'tdee': round(tdee, 1),
            'calorie_target': round(calorie_target, 1),
            'protein_target': round(protein_target, 1),
            'base_adherence': round(base_adherence, 2)
        })
    
    return users

def simulate_weekly_data(user, week_num, prev_weight):
    """Simulate one week of user data with realistic human behavior"""
    
    # Adherence varies by week (human nature - motivation fluctuates)
    # Week 1-2: High motivation (honeymoon phase)
    # Week 3-6: Dips (reality sets in)
    # Week 7-10: Stabilizes
    # Week 11-15: May slack or push harder (depends on results)
    
    if week_num <= 2:
        adherence_modifier = 1.1  # Extra motivated
    elif week_num <= 6:
        adherence_modifier = 0.85  # Motivation dips
    elif week_num <= 10:
        adherence_modifier = 1.0  # Steady
    else:
        adherence_modifier = np.random.choice([0.8, 1.05])  # Either slack or final push
    
    # Add random weekly variation (life happens - parties, stress, travel)
    weekly_chaos = np.random.uniform(0.85, 1.15)
    
    actual_adherence = user['base_adherence'] * adherence_modifier * weekly_chaos
    actual_adherence = np.clip(actual_adherence, 0.3, 1.0)  # Keep realistic
    
    # Actual calories eaten (influenced by adherence)
    if actual_adherence > 0.8:
        # Good week - close to target
        avg_daily_calories = user['calorie_target'] + np.random.randint(-100, 100)
    else:
        # Bad week - overeating or undereating
        avg_daily_calories = user['calorie_target'] + np.random.randint(-200, 400)
    
    # Actual gym days (influenced by adherence)
    actual_gym_days = int(user['gym_days_target'] * actual_adherence)
    actual_gym_days = max(0, min(7, actual_gym_days))  # Keep between 0-7
    
    # Actual protein (influenced by adherence)
    if actual_adherence > 0.75:
        avg_daily_protein = user['protein_target'] * np.random.uniform(0.9, 1.1)
    else:
        avg_daily_protein = user['protein_target'] * np.random.uniform(0.6, 0.9)
    
    # Calculate weight change using calorie math
    # 7700 calories deficit/surplus = 1 kg fat loss/gain
    weekly_calorie_diff = (avg_daily_calories - user['tdee']) * 7
    weekly_calorie_diff += actual_gym_days * 300  # Exercise burns ~300 cal/session
    
    expected_weight_change = weekly_calorie_diff / 7700  # kg
    
    # Add noise (water weight, measurement error, etc.)
    noise = np.random.uniform(-0.3, 0.3)
    actual_weight_change = expected_weight_change + noise
    
    # Update weight
    new_weight = prev_weight + actual_weight_change
    
    # Protein synthesis affects muscle (simplified)
    if user['goal'] == 'muscle_gain' and avg_daily_protein > user['protein_target'] * 0.8:
        muscle_gain_bonus = 0.1  # Small muscle gain
        new_weight += muscle_gain_bonus
    
    return {
        'week': week_num,
        'weight': round(new_weight, 1),
        'avg_daily_calories': round(avg_daily_calories, 1),
        'avg_daily_protein': round(avg_daily_protein, 1),
        'gym_days': actual_gym_days,
        'adherence_score': round(actual_adherence, 2),
        'weight_change': round(actual_weight_change, 2)
    }

def generate_full_dataset(n_users=90, n_weeks=15):
    """Generate complete FL training dataset"""
    
    users = generate_users(n_users)
    all_data = []
    
    for user in users:
        current_weight = user['start_weight']
        user_history = []
        
        for week in range(1, n_weeks + 1):
            week_data = simulate_weekly_data(user, week, current_weight)
            current_weight = week_data['weight']
            
            # Combine user profile + weekly data
            record = {
                **user,
                **week_data,
                'end_weight': current_weight
            }
            
            user_history.append(record)
            all_data.append(record)
        
        # Print summary for each user
        total_change = current_weight - user['start_weight']
        print(f"{user['user_id']}: {user['goal']}, "
              f"{user['start_weight']}kg → {current_weight:.1f}kg "
              f"({total_change:+.1f}kg over 15 weeks)")
    
    return pd.DataFrame(all_data)

# Generate the dataset
print("Generating 30 users with 15 weeks of data...\n")
df = generate_full_dataset(n_users=90, n_weeks=15)

# Save to CSV
df.to_csv('fl_training_data.csv', index=False)
print(f"\n✅ Generated {len(df)} records (90 users × 15 weeks)")
print(f"✅ Saved to 'fl_training_data.csv'")

# Show sample
print("\n📊 Sample data (first 5 rows):")
print(df.head())

# Show statistics
print("\n📈 Dataset Statistics:")
print(f"Weight loss users: {len(df[df['goal']=='weight_loss']['user_id'].unique())}")
print(f"Muscle gain users: {len(df[df['goal']=='muscle_gain']['user_id'].unique())}")
print(f"Maintenance users: {len(df[df['goal']=='maintenance']['user_id'].unique())}")
print(f"\nAverage adherence: {df['adherence_score'].mean():.2f}")
print(f"Average gym days/week: {df['gym_days'].mean():.1f}")

# Show adherence trends over weeks
print("\n📉 Adherence by Week (shows slacking pattern):")
adherence_by_week = df.groupby('week')['adherence_score'].mean()
for week, score in adherence_by_week.items():
    print(f"Week {week:2d}: {score:.2f}")