import numpy as np
from typing import Dict, List, Optional
import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.knowledge_base import *
from agent.utils import load_fl_model, EnsembleAdherencePredictor, EnsembleMacroRecommender


# PersonalizedFLModel class definition (needed to unpickle the pFL model)

class PersonalizedFLModel:
    def __init__(self, global_model, personalized_models, alpha=0.7):
        self.global_model = global_model
        self.personalized_models = personalized_models
        self.alpha = alpha  # 70% personal, 30% global

    def predict(self, X, user_id=None):
        global_pred = self.global_model.predict(X)
        if user_id and user_id in self.personalized_models:
            personal_pred = self.personalized_models[user_id].predict(X)
            return self.alpha * personal_pred + (1 - self.alpha) * global_pred
        return global_pred

    def predict_batch(self, X, user_ids):
        predictions = []
        for i, user_id in enumerate(user_ids):
            pred = self.predict(X[i:i+1], user_id)
            predictions.append(pred[0])
        return np.array(predictions)



# Helper to load pFL model (needs PersonalizedFLModel class in scope)

class EnsembleGlobalModel:
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)

def load_pfl_model(path: str):
    import __main__
    __main__.PersonalizedFLModel = PersonalizedFLModel
    __main__.EnsembleGlobalModel = EnsembleGlobalModel
    with open(path, 'rb') as f:
        return pickle.load(f)


class FitnessAgent:
    """
    Main AI Agent for Personalized Fitness Planning
    Uses Personalized Federated Learning (pFL) models for predictions.

    Architecture:
    1. PERCEIVE  – gather & validate user data
    2. REASON    – use pFL models + rules to make decisions
    3. PLAN      – generate meal & workout plans
    4. ACT       – return user-friendly recommendations
    5. ADAPT     – adjust plan based on weekly progress
    """

    def __init__(self, models_path='models/trained_models'):
        print(" Initializing Fitness Agent with pFL models...")

        # ── pFL Weight Prediction Model (BEST: 0.564 kg MAE) ──────────────────
        pfl_path = os.path.join(models_path, 'pfl_model.pkl')
        scaler_path = os.path.join(models_path, 'scaler.pkl')
        try:
            self.weight_model = load_pfl_model(pfl_path)
            with open(scaler_path, 'rb') as f:
                self.weight_scaler = pickle.load(f)
            self.use_pfl = True
            print("  ✅ pFL Weight Model loaded (MAE: 0.339 kg)")
        except Exception as e:
            print(f"  ⚠️  pFL model failed ({e}), falling back to basic model")
            self.weight_model = load_fl_model(os.path.join(models_path, 'weight_prediction_model.pkl'))
            self.weight_scaler = None
            self.use_pfl = False

        # ── Adherence Prediction Model (85.8% accuracy) ───────────────────────
        self.adherence_model = load_fl_model(os.path.join(models_path, 'adherence_prediction_model.pkl'))
        print("  ✅ Adherence Model loaded (86.7% accuracy)")

        # ── Macro Recommendation Model (86.5% R²) ─────────────────────────────
        self.macro_model = load_fl_model(os.path.join(models_path, 'macro_recommendation_model.pkl'))
        print("  ✅ Macro Model loaded (R²: 0.865)")

        print("🚀 Agent ready!\n")

    
    # 1. PERCEIVE
    
    def perceive(self, user_input: Dict) -> Dict:
        age            = int(user_input['age'])
        weight         = float(user_input['weight'])
        height         = float(user_input['height'])
        gender         = user_input['gender'].upper()
        goal           = user_input['goal'].lower()
        activity_level = user_input.get('activity_level', 'moderate').lower()
        gym_days       = int(user_input.get('gym_days', 3))

        bmi        = calculate_bmi(weight, height)
        bmr        = calculate_bmr(weight, height, age, gender)
        tdee       = calculate_tdee(bmr, activity_level)
        bf_est     = estimate_body_fat_percentage(bmi, age, gender)

        profile = {
            'age': age, 'weight': weight, 'height': height,
            'gender': gender, 'goal': goal,
            'activity_level': activity_level, 'gym_days': gym_days,
            'bmi': round(bmi, 1),
            'bmi_category': interpret_bmi(bmi),
            'bmr': round(bmr, 0),
            'tdee': round(tdee, 0),
            'estimated_body_fat': round(bf_est, 1)
        }

        print(f"👁️  Perceived: {age}yo {gender}, {weight}kg, Goal: {goal}")
        print(f"   BMI: {profile['bmi']} ({profile['bmi_category']}), TDEE: {profile['tdee']} kcal")
        return profile

    
    # 2. REASON
    
    def reason(self, user_profile: Dict, user_id: str = None) -> Dict:
        # Adherence prediction
        adherence_input = self._prepare_adherence_input(user_profile)
        predicted_adherence = float(np.clip(
            self.adherence_model.predict(adherence_input)[0], 0, 1
        ))
        print(f"🧠 FL Adherence: {predicted_adherence*100:.0f}%")

        # Plan difficulty
        if predicted_adherence >= ADHERENCE_THRESHOLDS['high']:
            plan_difficulty = 'aggressive'
            reasoning = "High adherence predicted — you can handle an aggressive plan"
        elif predicted_adherence >= ADHERENCE_THRESHOLDS['medium']:
            plan_difficulty = 'moderate'
            reasoning = "Moderate adherence predicted — balanced approach recommended"
        else:
            plan_difficulty = 'conservative'
            reasoning = "Lower adherence predicted — starting easy for sustainability"
        print(f"   Decision: {plan_difficulty.upper()} — {reasoning}")

        # Calorie target
        tdee = user_profile['tdee']
        goal = user_profile['goal']
        if goal == 'weight_loss':
            adjustment = CALORIE_ADJUSTMENTS['weight_loss'][plan_difficulty]
        elif goal == 'muscle_gain':
            adjustment = CALORIE_ADJUSTMENTS['muscle_gain']['moderate_bulk']
        else:
            adjustment = CALORIE_ADJUSTMENTS['maintenance']['default']

        calorie_target, warning = validate_calorie_target(
            tdee + adjustment, user_profile['gender'], goal
        )
        print(f"   Calories: {calorie_target:.0f}/day (TDEE {tdee:.0f} {adjustment:+})")

        # Macro recommendation
        macro_input = self._prepare_macro_input(user_profile, calorie_target)
        macros = self.macro_model.recommend_macros(macro_input)
        print(f"   Macros → P:{macros['protein_g']}g C:{macros['carbs_g']}g F:{macros['fat_g']}g")

        # Workout template
        workout_template = select_workout_template(user_profile['gym_days'])

        # pFL weight prediction for week 1 target
        weight_input_raw = np.array([[
            user_profile['weight'],
            calorie_target,
            macros['protein_g'],
            user_profile['gym_days'],
            user_profile['age'],
            1 if user_profile['gender'] == 'M' else 0,
            user_profile['tdee']
        ]])
        predicted_week1_weight = self._predict_weight(weight_input_raw, user_id)
        expected_change = round(predicted_week1_weight - user_profile['weight'], 2)
        print(f"   pFL Week-1 Prediction: {predicted_week1_weight:.2f} kg ({expected_change:+.2f} kg)")

        return {
            'predicted_adherence':     round(predicted_adherence, 2),
            'plan_difficulty':         plan_difficulty,
            'reasoning':               reasoning,
            'calorie_target':          round(calorie_target, 0),
            'macros':                  macros,
            'workout_template':        workout_template,
            'warning':                 warning,
            'predicted_week1_weight':  round(predicted_week1_weight, 2),
            'expected_weekly_change':  expected_change
        }

    
    # 3. PLAN
    
    def plan(self, user_profile: Dict, decisions: Dict) -> Dict:
        meal_plan        = self._generate_meal_plan(
            decisions['calorie_target'], decisions['macros'], user_profile['goal']
        )
        workout_schedule = decisions['workout_template']['routine']
        water_intake     = calculate_water_intake(user_profile['weight'], user_profile['activity_level'])

        goal = user_profile['goal']
        if goal == 'weight_loss':
            weekly_target     = -0.5 if decisions['plan_difficulty'] == 'moderate' else -0.7
            target_description = f"Lose {abs(weekly_target)} kg per week"
        elif goal == 'muscle_gain':
            weekly_target      = 0.3
            target_description = "Gain 0.3 kg per week (lean mass)"
        else:
            weekly_target      = 0
            target_description = "Maintain current weight"

        print(f"📋 Plan: {len(meal_plan)} meals/day | "
              f"{len(decisions['workout_template']['days'])} workout days | "
              f"{water_intake}L water")

        return {
            'meal_plan':           meal_plan,
            'workout_schedule':    workout_schedule,
            'workout_days':        decisions['workout_template']['days'],
            'water_intake_liters': water_intake,
            'weekly_weight_target': weekly_target,
            'target_description':  target_description,
            'rest_days':           7 - len(decisions['workout_template']['days'])
        }

    
    # 4. ACT
    
    def act(self, user_profile: Dict, decisions: Dict, plan: Dict) -> Dict:
        return {
            'user_profile': {
                'bmi':                user_profile['bmi'],
                'bmi_category':       user_profile['bmi_category'],
                'tdee':               user_profile['tdee'],
                'estimated_body_fat': user_profile['estimated_body_fat']
            },
            'plan_overview': {
                'goal':           user_profile['goal'],
                'difficulty':     decisions['plan_difficulty'],
                'daily_calories': decisions['calorie_target'],
                'macros':         decisions['macros'],
                'reasoning':      decisions['reasoning'],
                'fl_prediction': {
                    'model_used': 'Personalized FL (pFL)' if self.use_pfl else 'Basic FL',
                    'model_mae': '0.339 kg' if self.use_pfl else 'N/A',
                    'predicted_week1_weight': decisions['predicted_week1_weight'],
                    'expected_weekly_change': decisions['expected_weekly_change']
                }
            },
            'meal_plan': plan['meal_plan'],
            'workout_schedule': {
                'template_name': decisions['workout_template']['name'],
                'days':          plan['workout_days'],
                'routine':       plan['workout_schedule']
            },
            'weekly_targets': {
                'weight_change': plan['weekly_weight_target'],
                'description':   plan['target_description'],
                'water_intake':  plan['water_intake_liters']
            },
            'success_tips': self._generate_success_tips(user_profile['goal'])
        }

    
    # 5. ADAPT  (called weekly from /api/user/<id>/progress)
    
    def adapt(self, user_profile: Dict, progress_data: Dict,
              current_plan: Dict, user_id: str = None) -> Dict:

        weight_input_raw = self._prepare_weight_input(user_profile, progress_data)
        predicted_weight = self._predict_weight(weight_input_raw, user_id)

        actual_weight = progress_data['current_weight']
        difference    = actual_weight - predicted_weight
        current_cal   = current_plan['plan_overview']['daily_calories']

        print(f"🔄 Adapt → Predicted: {predicted_weight:.1f} kg | "
              f"Actual: {actual_weight:.1f} kg | Diff: {difference:+.2f} kg")

        if abs(difference) < 0.5:
            status  = 'on_track'
            message = "Great job! You're right on track. Continue with current plan."
            cal_adj = 0
        elif difference > 0.5:
            status  = 'adjust_down'
            message = "Progress slower than predicted. Reducing calories by 100."
            cal_adj = -100
        else:
            status  = 'adjust_up'
            message = "Progress faster than predicted. Increasing calories for sustainability."
            cal_adj = +100

        new_cal = current_cal + cal_adj
        print(f"   Status: {status} | Calories: {current_cal} → {new_cal}")

        return {
            'status':             status,
            'message':            message,
            'calorie_adjustment': cal_adj,
            'new_calorie_target': new_cal,
            'predicted_weight':   round(predicted_weight, 2),
            'actual_weight':      actual_weight,
            'difference':         round(difference, 2),
            'model_used':         'pFL (0.564 kg MAE)' if self.use_pfl else 'Basic FL'
        }

    
    # PUBLIC ENTRY POINT
    
    def create_personalized_plan(self, user_input: Dict, user_id: str = None) -> Dict:
        """Main function called by Flask on signup / plan creation."""
        print("\n" + "="*50)
        print("🏋️  CREATING PERSONALIZED FITNESS PLAN")
        print("="*50)

        user_profile = self.perceive(user_input)
        decisions    = self.reason(user_profile, user_id)
        plan         = self.plan(user_profile, decisions)
        recommendation = self.act(user_profile, decisions, plan)

        print("✅ PLAN CREATED!\n")
        return recommendation

    
    # PRIVATE HELPERS
    
    def _predict_weight(self, X_raw: np.ndarray, user_id: str = None) -> float:
        """Use pFL model with Differential Privacy noise (ε=5.0)."""
        if self.use_pfl and self.weight_scaler:
            X_scaled = self.weight_scaler.transform(X_raw)
            prediction = float(self.weight_model.predict(X_scaled, user_id)[0])
        else:
            prediction = float(self.weight_model.predict(X_raw)[0])

            # Apply Differential Privacy — Laplace mechanism (ε=5.0)
        dp_noise  = np.random.laplace(0, 1.0 / 5.0)
        prediction = round(prediction + dp_noise, 2)
        return prediction

    def _prepare_adherence_input(self, user_profile: Dict) -> np.ndarray:
        tdee = user_profile['tdee']
        calorie_target = tdee - 300  # approximate for initial prediction
        calorie_deficit_pct = (tdee - calorie_target) / tdee

        return np.array([[
            user_profile['age'],
            1 if user_profile['gender'] == 'M' else 0,
            user_profile['weight'],
            user_profile['gym_days'],
            calorie_deficit_pct,
            1 if user_profile['goal'] == 'weight_loss' else 0,
            1 if user_profile['goal'] == 'muscle_gain' else 0,
            1 if user_profile['activity_level'] == 'sedentary' else 0,
            1 if user_profile['activity_level'] == 'active' else 0,
            1  # week 1
        ]])

    def _prepare_macro_input(self, user_profile: Dict, calorie_target: float) -> np.ndarray:
        return np.array([
            user_profile['age'],
            1 if user_profile['gender'] == 'M' else 0,
            user_profile['weight'],
            user_profile['weight'],
            1 if user_profile['goal'] == 'weight_loss' else 0,
            1 if user_profile['goal'] == 'muscle_gain' else 0,
            1 if user_profile['goal'] == 'maintenance' else 0,
            user_profile['gym_days'],
            1 if user_profile['activity_level'] == 'sedentary' else 0,
            1 if user_profile['activity_level'] == 'light' else 0,
            1 if user_profile['activity_level'] == 'moderate' else 0,
            1 if user_profile['activity_level'] == 'active' else 0,
            calorie_target
        ])

    def _prepare_weight_input(self, user_profile: Dict, progress_data: Dict) -> np.ndarray:
        return np.array([[
            progress_data.get('last_weight', user_profile['weight']),
            progress_data.get('avg_calories', 2000),
            progress_data.get('avg_protein', 120),
            progress_data.get('workouts_done', 3),
            user_profile['age'],
            1 if user_profile['gender'] == 'M' else 0,
            user_profile['tdee']
        ]])

    def _generate_meal_plan(self, calories: float, macros: Dict, goal: str) -> List[Dict]:
        return [
            {
                'meal': 'Breakfast',
                'calories': round(calories * 0.30),
                'protein': round(macros['protein_g'] * 0.30),
                'suggestions': ['Poha', 'Upma', 'Idli with Sambar', 'Paratha with Curd',
                                 'Oats with Banana', 'Egg Bhurji with Toast']
            },
            {
                'meal': 'Lunch',
                'calories': round(calories * 0.40),
                'protein': round(macros['protein_g'] * 0.40),
                'suggestions': ['Dal Bhat', 'Rajma Chawal', 'Chicken Curry with Rice',
                                 'Vegetarian Thali', 'Chole with Roti', 'Paneer Sabzi with Rice']
            },
            {
                'meal': 'Dinner',
                'calories': round(calories * 0.30),
                'protein': round(macros['protein_g'] * 0.30),
                'suggestions': ['Roti with Dal', 'Grilled Chicken with Veggies',
                                 'Fish Curry with Rice', 'Paneer Sabzi with Roti',
                                 'Moong Dal Khichdi', 'Egg Curry with Brown Rice']
            }
        ]

    def _generate_success_tips(self, goal: str) -> List[str]:
        common = [
            "Track your meals consistently",
            "Stay hydrated — drink water throughout the day",
            "Get 7-8 hours of sleep for recovery"
        ]
        specific = {
            'weight_loss': [
                "Eat protein with every meal to stay full",
                "Focus on whole foods over processed",
                "Don't skip meals — it leads to overeating later"
            ],
            'muscle_gain': [
                "Eat within 2 hours after workout",
                "Progressive overload — increase weights gradually",
                "Be patient — muscle building takes time"
            ],
            'maintenance': [
                "Listen to your body's hunger cues",
                "Stay active with activities you enjoy",
                "80/20 rule — be flexible, not perfect"
            ]
        }
        return common + specific.get(goal, [])