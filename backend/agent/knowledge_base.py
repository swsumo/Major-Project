# Calories per gram of macronutrient
CALORIES_PER_GRAM = {
    'protein': 4,
    'carbs': 4,
    'fat': 9
}

# Protein requirements (grams per kg bodyweight)
PROTEIN_REQUIREMENTS = {
    'weight_loss': {
        'min': 1.6,
        'optimal': 1.8,
        'max': 2.2
    },
    'muscle_gain': {
        'min': 1.8,
        'optimal': 2.0,
        'max': 2.4
    },
    'maintenance': {
        'min': 1.0,
        'optimal': 1.2,
        'max': 1.6
    }
}

SAFETY_LIMITS = {
    'min_calories_male': 1500,
    'min_calories_female': 1200,
    'max_calorie_deficit': 1000,  # calories per day
    'max_weight_loss_per_week': 1.0,  # kg
    'max_weight_gain_per_week': 0.5,  # kg (mostly muscle)
    'min_rest_days': 1,  # per week
    'max_gym_days': 6  # per week
}

CALORIE_ADJUSTMENTS = {
    'weight_loss': {
        'aggressive': -500,  # Fast results, harder to follow
        'moderate': -300,    # Balanced approach
        'conservative': -200  # Slow and steady
    },
    'muscle_gain': {
        'lean_bulk': 200,    # Minimize fat gain
        'moderate_bulk': 300,  # Balanced
        'aggressive_bulk': 500  # Fast gains, more fat
    },
    'maintenance': {
        'default': 0
    }
}

ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,      # Little to no exercise
    'light': 1.375,        # 1-3 days per week
    'moderate': 1.55,      # 3-5 days per week
    'active': 1.725,       # 6-7 days per week
    'very_active': 1.9     # Athlete level
}

WORKOUT_TEMPLATES = {
    'beginner_3day': {
        'name': '3-Day Full Body (Beginner)',
        'days': ['Monday', 'Wednesday', 'Friday'],
        'routine': {
            'Monday': ['Squats 3x10', 'Push-ups 3x8', 'Rows 3x10', 'Plank 3x30s'],
            'Wednesday': ['Deadlifts 3x8', 'Bench Press 3x8', 'Lat Pulldown 3x10', 'Crunches 3x15'],
            'Friday': ['Lunges 3x10', 'Shoulder Press 3x10', 'Pull-ups 3x6', 'Leg Raises 3x12']
        }
    },
    'intermediate_4day': {
        'name': '4-Day Upper/Lower Split',
        'days': ['Monday', 'Tuesday', 'Thursday', 'Friday'],
        'routine': {
            'Monday': ['Bench Press 4x8', 'Rows 4x10', 'Shoulder Press 3x10', 'Bicep Curls 3x12'],
            'Tuesday': ['Squats 4x8', 'Deadlifts 3x6', 'Leg Press 3x12', 'Calf Raises 4x15'],
            'Thursday': ['Incline Press 4x10', 'Pull-ups 4x8', 'Lateral Raises 3x12', 'Tricep Dips 3x10'],
            'Friday': ['Front Squats 3x10', 'Romanian Deadlifts 3x10', 'Leg Curls 3x12', 'Planks 3x45s']
        }
    },
    'advanced_5day': {
        'name': '5-Day Push/Pull/Legs',
        'days': ['Monday', 'Tuesday', 'Wednesday', 'Friday', 'Saturday'],
        'routine': {
            'Monday': ['Bench Press 5x5', 'Incline DB Press 4x10', 'Shoulder Press 4x8', 'Tricep Extensions 3x12'],
            'Tuesday': ['Deadlifts 5x5', 'Pull-ups 4x8', 'Barbell Rows 4x10', 'Bicep Curls 3x12'],
            'Wednesday': ['Squats 5x5', 'Leg Press 4x12', 'Leg Curls 3x12', 'Calf Raises 4x15'],
            'Friday': ['Overhead Press 4x8', 'Incline Bench 4x10', 'Dips 3x12', 'Lateral Raises 3x15'],
            'Saturday': ['Romanian Deadlifts 4x10', 'Lunges 3x12', 'Leg Extensions 3x15', 'Abs Circuit 4 rounds']
        }
    }
}

ADHERENCE_THRESHOLDS = {
    'high': 0.75,      # User likely to stick to plan
    'medium': 0.5,     # May struggle, needs easier plan
    'low': 0.3         # Very likely to quit
}


def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    """
    Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation
    
    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters
        age: Age in years
        gender: 'M' or 'F'
    
    Returns:
        BMR in calories
    """
    if gender.upper() == 'M':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:  # Female
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    
    return bmr

def calculate_tdee(bmr: float, activity_level: str) -> float:
    """
    Calculate Total Daily Energy Expenditure
    
    Args:
        bmr: Basal Metabolic Rate
        activity_level: One of: sedentary, light, moderate, active, very_active
    
    Returns:
        TDEE in calories
    """
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level.lower(), 1.2)
    return bmr * multiplier

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """
    Calculate Body Mass Index
    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters
    
    Returns:
        BMI value
    """
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)

def interpret_bmi(bmi: float) -> str:
    """
    Interpret BMI value
    Returns:
        Category string
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def estimate_body_fat_percentage(bmi: float, age: int, gender: str) -> float:
    """
    Rough estimate of body fat % using BMI
    (Not very accurate, but gives a ballpark)
    Returns:
        Estimated body fat percentage
    """
    if gender.upper() == 'M':
        bf = (1.20 * bmi) + (0.23 * age) - 16.2
    else:  # Female
        bf = (1.20 * bmi) + (0.23 * age) - 5.4
    
    return max(5, min(50, bf))  # Clamp between 5-50%

def select_workout_template(gym_days: int, experience_level: str = 'beginner') -> dict:
    """
    Select appropriate workout template based on availability and experience
    Args:
        gym_days: Days per week user can train (1-7)
        experience_level: beginner, intermediate, advanced
    
    Returns:
        Workout template dict
    """
    if gym_days <= 3:
        return WORKOUT_TEMPLATES['beginner_3day']
    elif gym_days == 4:
        return WORKOUT_TEMPLATES['intermediate_4day']
    else:  # 5+ days
        return WORKOUT_TEMPLATES['advanced_5day']

def calculate_protein_target(weight_kg: float, goal: str) -> float:
    """
    Calculate optimal protein intake
    Args:
        weight_kg: Body weight in kg
        goal: weight_loss, muscle_gain, or maintenance
    Returns:
        Protein target in grams
    """
    multiplier = PROTEIN_REQUIREMENTS[goal]['optimal']
    return weight_kg * multiplier

def validate_calorie_target(calories: float, gender: str, goal: str) -> tuple:
    """
    Validate and adjust calorie target for safety
    Returns:
        (adjusted_calories, warning_message or None)
    """
    min_cal = SAFETY_LIMITS['min_calories_male'] if gender.upper() == 'M' else SAFETY_LIMITS['min_calories_female']
    
    if calories < min_cal:
        return min_cal, f"Calorie target too low. Increased to minimum safe level ({min_cal} cal)"
    
    return calories, None


MEAL_TIMING = {
    'weight_loss': {
        'meals_per_day': 3,
        'snacks': 1,
        'timing': 'Regular meals, avoid late-night eating'
    },
    'muscle_gain': {
        'meals_per_day': 4,
        'snacks': 2,
        'timing': 'Frequent meals, protein within 2h post-workout'
    },
    'maintenance': {
        'meals_per_day': 3,
        'snacks': 1,
        'timing': 'Flexible, based on preference'
    }
}
def calculate_water_intake(weight_kg: float, activity_level: str) -> float:
    """
    Calculate daily water intake recommendation
    
    Returns:
        Water in liters
    """
    base = weight_kg * 0.033  
    
    if activity_level in ['active', 'very_active']:
        base *= 1.2  # Extra for sweating
    
    return round(base, 1)