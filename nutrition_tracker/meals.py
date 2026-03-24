import json
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
from dataclasses import dataclass

class PortionSize(Enum):
    LIGHT = "Light"
    REGULAR = "Regular"
    HEAVY = "Heavy"

class MealType(Enum):
    BREAKFAST = "Breakfast"
    LUNCH = "Lunch"
    DINNER = "Dinner"
    SNACK = "Snack"

@dataclass
class NutritionalInfo:
    """Data class for nutritional information"""
    calories: float
    protein: float  # in grams
    carbs: float    # in grams
    fat: float      # in grams
    
    def __add__(self, other: 'NutritionalInfo') -> 'NutritionalInfo':
        return NutritionalInfo(
            calories=self.calories + other.calories,
            protein=self.protein + other.protein,
            carbs=self.carbs + other.carbs,
            fat=self.fat + other.fat
        )
    
    def __mul__(self, factor: float) -> 'NutritionalInfo':
        return NutritionalInfo(
            calories=self.calories * factor,
            protein=self.protein * factor,
            carbs=self.carbs * factor,
            fat=self.fat * factor
        )

class IndianMealLogger:
    """Main class for Indian meal logging and tracking"""
    
    # MEAL_COMPONENTS - Base ingredients for detailed logging
    MEAL_COMPONENTS: Dict[str, NutritionalInfo] = {
        # Rice and Grains (per cup cooked)
        "rice_white_cooked": NutritionalInfo(205, 4.3, 44.5, 0.4),
        "rice_brown_cooked": NutritionalInfo(216, 5.0, 45.0, 1.8),
        "rice_basmati_cooked": NutritionalInfo(210, 4.5, 45.0, 0.5),
        
        # Daals/Pulses (per katori ~100g cooked)
        "daal_yellow": NutritionalInfo(105, 7.0, 17.0, 1.0),
        "daal_toor": NutritionalInfo(100, 6.5, 18.0, 0.5),
        "daal_moong": NutritionalInfo(106, 7.0, 18.0, 0.5),
        "daal_masoor": NutritionalInfo(116, 9.0, 20.0, 0.5),
        "daal_chana": NutritionalInfo(120, 7.0, 20.0, 1.0),
        "daal_urad": NutritionalInfo(130, 8.0, 22.0, 1.0),
        
        # Breads (per piece)
        "roti_whole_wheat": NutritionalInfo(70, 3.0, 15.0, 0.5),
        "roti_multi_grain": NutritionalInfo(75, 3.5, 14.0, 1.0),
        "naan_plain": NutritionalInfo(320, 9.0, 56.0, 6.0),
        "paratha_plain": NutritionalInfo(150, 4.0, 20.0, 6.0),
        "paratha_aloo": NutritionalInfo(200, 5.0, 25.0, 8.0),
        "chapati": NutritionalInfo(60, 2.0, 12.0, 0.5),
        "bhatura": NutritionalInfo(250, 6.0, 38.0, 8.0),
        "puri": NutritionalInfo(80, 2.0, 10.0, 4.0),
        
        # Vegetables (per serving ~100g)
        "aloo_gobi": NutritionalInfo(120, 3.0, 15.0, 5.0),
        "bhindi_masala": NutritionalInfo(90, 3.0, 12.0, 4.0),
        "baingan_bharta": NutritionalInfo(110, 3.5, 10.0, 6.0),
        "palak_paneer": NutritionalInfo(180, 10.0, 8.0, 12.0),
        "mutter_paneer": NutritionalInfo(200, 11.0, 12.0, 13.0),
        "chana_masala": NutritionalInfo(160, 6.0, 25.0, 5.0),
        "rajma_masala": NutritionalInfo(170, 8.0, 25.0, 5.0),
        "mixed_vegetables": NutritionalInfo(80, 2.5, 12.0, 2.5),
        "dal_fry": NutritionalInfo(140, 7.0, 18.0, 4.0),
        
        # Protein Sources
        "paneer_100g": NutritionalInfo(265, 18.0, 4.0, 20.0),
        "chicken_curry": NutritionalInfo(200, 20.0, 5.0, 12.0),
        "fish_curry": NutritionalInfo(180, 22.0, 3.0, 9.0),
        "egg_curry": NutritionalInfo(150, 12.0, 3.0, 10.0),
        
        # Breakfast Items
        "idli": NutritionalInfo(40, 2.0, 8.0, 0.5),
        "dosa_plain": NutritionalInfo(125, 3.0, 20.0, 3.5),
        "dosa_masala": NutritionalInfo(200, 6.0, 25.0, 8.0),
        "poha": NutritionalInfo(150, 3.0, 30.0, 3.0),
        "upma": NutritionalInfo(160, 4.0, 25.0, 5.0),
        
        # Condiments and Sides
        "sambar_cup": NutritionalInfo(80, 4.0, 12.0, 2.0),
        "chutney_coconut": NutritionalInfo(50, 1.0, 4.0, 3.5),
        "chutney_tomato": NutritionalInfo(30, 1.0, 5.0, 1.0),
        "curd_cup": NutritionalInfo(150, 8.0, 11.0, 8.0),
        "raita_plain": NutritionalInfo(100, 4.0, 8.0, 6.0),
        
        # Fats and Oils (per tsp)
        "ghee": NutritionalInfo(45, 0.0, 0.0, 5.0),
        "oil": NutritionalInfo(40, 0.0, 0.0, 4.5),
        "butter": NutritionalInfo(35, 0.0, 0.0, 4.0),
    }
    
    # COMPLETE MEALS with portion sizes
    MEALS: Dict[str, Dict[str, Union[Dict[str, NutritionalInfo], List[str], str]]] = {
        # Rice-based meals
        "daal_bhat": {
            "name": "Daal Bhat (Rice + Lentils)",
            "description": "Steamed rice with lentil curry",
            "category": MealType.LUNCH.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(350, 12.0, 60.0, 6.0),
                PortionSize.REGULAR.value: NutritionalInfo(450, 15.0, 75.0, 8.0),
                PortionSize.HEAVY.value: NutritionalInfo(550, 18.0, 90.0, 10.0)
            },
            "components": ["rice_basmati_cooked", "daal_yellow", "ghee"]
        },
        
        "rajma_chawal": {
            "name": "Rajma Chawal",
            "description": "Kidney beans curry with rice",
            "category": MealType.LUNCH.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(380, 15.0, 65.0, 7.0),
                PortionSize.REGULAR.value: NutritionalInfo(480, 18.0, 80.0, 9.0),
                PortionSize.HEAVY.value: NutritionalInfo(580, 21.0, 95.0, 11.0)
            },
            "components": ["rice_basmati_cooked", "rajma_masala"]
        },
        
        "biryani_veg": {
            "name": "Vegetable Biryani",
            "description": "Fragrant rice dish with vegetables and spices",
            "category": MealType.LUNCH.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(400, 10.0, 70.0, 10.0),
                PortionSize.REGULAR.value: NutritionalInfo(550, 13.0, 90.0, 15.0),
                PortionSize.HEAVY.value: NutritionalInfo(700, 16.0, 110.0, 20.0)
            },
            "components": ["rice_basmati_cooked", "mixed_vegetables", "oil"]
        },
        
        "biryani_chicken": {
            "name": "Chicken Biryani",
            "description": "Aromatic rice dish with chicken",
            "category": MealType.LUNCH.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(450, 25.0, 60.0, 15.0),
                PortionSize.REGULAR.value: NutritionalInfo(600, 32.0, 75.0, 20.0),
                PortionSize.HEAVY.value: NutritionalInfo(750, 38.0, 90.0, 25.0)
            },
            "components": ["rice_basmati_cooked", "chicken_curry", "ghee"]
        },
        
        # Bread-based meals
        "roti_sabzi": {
            "name": "Roti with Sabzi",
            "description": "Whole wheat bread with vegetable curry",
            "category": MealType.DINNER.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(300, 10.0, 45.0, 8.0),
                PortionSize.REGULAR.value: NutritionalInfo(400, 13.0, 60.0, 10.0),
                PortionSize.HEAVY.value: NutritionalInfo(500, 16.0, 75.0, 12.0)
            },
            "components": ["roti_whole_wheat", "mixed_vegetables"]
        },
        
        "paratha_curd": {
            "name": "Paratha with Curd",
            "description": "Stuffed flatbread with yogurt",
            "category": MealType.BREAKFAST.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(350, 10.0, 40.0, 15.0),
                PortionSize.REGULAR.value: NutritionalInfo(450, 13.0, 50.0, 20.0),
                PortionSize.HEAVY.value: NutritionalInfo(550, 16.0, 60.0, 25.0)
            },
            "components": ["paratha_aloo", "curd_cup", "ghee"]
        },
        
        "chole_bhature": {
            "name": "Chole Bhature",
            "description": "Chickpea curry with fried bread",
            "category": MealType.BREAKFAST.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(500, 15.0, 70.0, 18.0),
                PortionSize.REGULAR.value: NutritionalInfo(650, 18.0, 85.0, 25.0),
                PortionSize.HEAVY.value: NutritionalInfo(800, 21.0, 100.0, 32.0)
            },
            "components": ["chana_masala", "bhatura", "oil"]
        },
        
        # Breakfast items
        "idli_sambar": {
            "name": "Idli with Sambar",
            "description": "Steamed rice cakes with lentil soup",
            "category": MealType.BREAKFAST.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(200, 8.0, 35.0, 3.0),
                PortionSize.REGULAR.value: NutritionalInfo(300, 12.0, 50.0, 4.0),
                PortionSize.HEAVY.value: NutritionalInfo(400, 16.0, 65.0, 5.0)
            },
            "components": ["idli", "sambar_cup", "chutney_coconut"]
        },
        
        "dosa_chutney": {
            "name": "Dosa with Chutney",
            "description": "Crispy fermented crepe with coconut chutney",
            "category": MealType.BREAKFAST.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(250, 5.0, 40.0, 7.0),
                PortionSize.REGULAR.value: NutritionalInfo(350, 7.0, 55.0, 10.0),
                PortionSize.HEAVY.value: NutritionalInfo(450, 9.0, 70.0, 13.0)
            },
            "components": ["dosa_masala", "chutney_coconut", "sambar_cup"]
        },
        
        "poha": {
            "name": "Poha",
            "description": "Flattened rice with peanuts and vegetables",
            "category": MealType.BREAKFAST.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(200, 4.0, 35.0, 5.0),
                PortionSize.REGULAR.value: NutritionalInfo(250, 5.0, 45.0, 6.0),
                PortionSize.HEAVY.value: NutritionalInfo(300, 6.0, 55.0, 7.0)
            },
            "components": ["poha", "oil"]
        },
        
        "upma": {
            "name": "Upma",
            "description": "Semolina porridge with vegetables",
            "category": MealType.BREAKFAST.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(220, 5.0, 35.0, 6.0),
                PortionSize.REGULAR.value: NutritionalInfo(280, 6.0, 45.0, 8.0),
                PortionSize.HEAVY.value: NutritionalInfo(340, 7.0, 55.0, 10.0)
            },
            "components": ["upma", "oil"]
        },
        
        # North Indian Specialties
        "paneer_tikka_masala": {
            "name": "Paneer Tikka Masala",
            "description": "Grilled cottage cheese in creamy tomato gravy",
            "category": MealType.DINNER.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(350, 18.0, 20.0, 22.0),
                PortionSize.REGULAR.value: NutritionalInfo(450, 22.0, 25.0, 28.0),
                PortionSize.HEAVY.value: NutritionalInfo(550, 26.0, 30.0, 34.0)
            },
            "components": ["paneer_100g", "ghee", "oil"]
        },
        
        "butter_chicken": {
            "name": "Butter Chicken",
            "description": "Chicken in creamy tomato gravy",
            "category": MealType.DINNER.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(380, 25.0, 15.0, 25.0),
                PortionSize.REGULAR.value: NutritionalInfo(500, 30.0, 20.0, 32.0),
                PortionSize.HEAVY.value: NutritionalInfo(620, 35.0, 25.0, 39.0)
            },
            "components": ["chicken_curry", "butter", "ghee"]
        },
        
        "palak_paneer_combo": {
            "name": "Palak Paneer with Roti",
            "description": "Spinach and cottage cheese with Indian bread",
            "category": MealType.DINNER.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(400, 18.0, 35.0, 20.0),
                PortionSize.REGULAR.value: NutritionalInfo(500, 22.0, 45.0, 25.0),
                PortionSize.HEAVY.value: NutritionalInfo(600, 26.0, 55.0, 30.0)
            },
            "components": ["palak_paneer", "roti_whole_wheat", "ghee"]
        },
        
        # South Indian Specialties
        "vada_sambar": {
            "name": "Medu Vada with Sambar",
            "description": "Lentil doughnuts with lentil soup",
            "category": MealType.BREAKFAST.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(280, 8.0, 40.0, 10.0),
                PortionSize.REGULAR.value: NutritionalInfo(350, 10.0, 50.0, 13.0),
                PortionSize.HEAVY.value: NutritionalInfo(420, 12.0, 60.0, 16.0)
            },
            "components": ["sambar_cup", "oil"]
        },
        
        "uttapam": {
            "name": "Uttapam",
            "description": "Thick pancake with vegetables",
            "category": MealType.BREAKFAST.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(180, 6.0, 30.0, 4.0),
                PortionSize.REGULAR.value: NutritionalInfo(240, 8.0, 40.0, 5.0),
                PortionSize.HEAVY.value: NutritionalInfo(300, 10.0, 50.0, 6.0)
            },
            "components": ["chutney_coconut", "sambar_cup"]
        },
        
        # Street Food
        "pav_bhaji": {
            "name": "Pav Bhaji",
            "description": "Spicy vegetable mash with buttered buns",
            "category": MealType.SNACK.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(350, 8.0, 45.0, 15.0),
                PortionSize.REGULAR.value: NutritionalInfo(450, 10.0, 55.0, 20.0),
                PortionSize.HEAVY.value: NutritionalInfo(550, 12.0, 65.0, 25.0)
            },
            "components": ["butter", "mixed_vegetables"]
        },
        
        "samosa": {
            "name": "Samosa (2 pieces)",
            "description": "Fried pastry with potato filling",
            "category": MealType.SNACK.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(250, 4.0, 30.0, 12.0),
                PortionSize.REGULAR.value: NutritionalInfo(350, 5.0, 40.0, 18.0),
                PortionSize.HEAVY.value: NutritionalInfo(450, 6.0, 50.0, 24.0)
            },
            "components": ["oil"]
        },
        
        # More complete meals
        "thali_veg": {
            "name": "Vegetarian Thali",
            "description": "Complete meal with rice, roti, dal, sabzi, curd",
            "category": MealType.LUNCH.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(600, 22.0, 90.0, 18.0),
                PortionSize.REGULAR.value: NutritionalInfo(750, 27.0, 110.0, 23.0),
                PortionSize.HEAVY.value: NutritionalInfo(900, 32.0, 130.0, 28.0)
            },
            "components": ["rice_basmati_cooked", "roti_whole_wheat", "daal_yellow", "mixed_vegetables", "curd_cup"]
        },
        
        "thali_nonveg": {
            "name": "Non-Vegetarian Thali",
            "description": "Complete meal with rice, roti, chicken, dal, curd",
            "category": MealType.DINNER.value,
            "portions": {
                PortionSize.LIGHT.value: NutritionalInfo(700, 35.0, 80.0, 25.0),
                PortionSize.REGULAR.value: NutritionalInfo(850, 42.0, 95.0, 32.0),
                PortionSize.HEAVY.value: NutritionalInfo(1000, 49.0, 110.0, 39.0)
            },
            "components": ["rice_basmati_cooked", "roti_whole_wheat", "chicken_curry", "daal_yellow", "curd_cup"]
        }
    }
    
    def __init__(self):
        """Initialize the meal logger"""
        self.logged_meals = []
        
    def get_meal_options(self, meal_name: str) -> Optional[Dict]:
        """Get available options for a specific meal"""
        meal_key = meal_name.lower().replace(" ", "_")
        if meal_key in self.MEALS:
            return self.MEALS[meal_key]
        return None
    
    def log_meal(self, meal_name: str, portion_size: str, 
                 custom_components: Optional[List[Tuple[str, float]]] = None) -> Dict:
        """
        Log a meal with portion size or custom components
        
        Args:
            meal_name: Name of the meal
            portion_size: Light, Regular, or Heavy
            custom_components: List of (component_name, multiplier) for detailed logging
        
        Returns:
            Dictionary with meal details and nutritional info
        """
        meal_key = meal_name.lower().replace(" ", "_")
        
        if meal_key not in self.MEALS:
            raise ValueError(f"Meal '{meal_name}' not found in database")
        
        meal_data = self.MEALS[meal_key]
        
        if custom_components:
            # Detailed component-based logging
            total_nutrition = NutritionalInfo(0, 0, 0, 0)
            components_detail = []
            
            for component, multiplier in custom_components:
                if component in self.MEAL_COMPONENTS:
                    nutrition = self.MEAL_COMPONENTS[component] * multiplier
                    total_nutrition += nutrition
                    components_detail.append({
                        "component": component,
                        "multiplier": multiplier,
                        "nutrition": nutrition
                    })
                else:
                    print(f"Warning: Component '{component}' not found")
            
            meal_log = {
                "meal_name": meal_data["name"],
                "meal_key": meal_key,
                "type": "custom_components",
                "portion_size": portion_size,
                "components": components_detail,
                "nutrition": total_nutrition,
                "timestamp": self._get_timestamp()
            }
            
        else:
            # Quick logging with portion sizes
            if portion_size not in meal_data["portions"]:
                raise ValueError(f"Portion size '{portion_size}' not available. Choose from {list(meal_data['portions'].keys())}")
            
            meal_log = {
                "meal_name": meal_data["name"],
                "meal_key": meal_key,
                "type": "quick_log",
                "portion_size": portion_size,
                "nutrition": meal_data["portions"][portion_size],
                "category": meal_data["category"],
                "description": meal_data.get("description", ""),
                "timestamp": self._get_timestamp()
            }
        
        self.logged_meals.append(meal_log)
        return meal_log
    
    def suggest_meal(self, calorie_target: float = 500, 
                     protein_target: float = 20) -> List[Dict]:
        """
        Suggest meals based on calorie and protein targets
        
        Args:
            calorie_target: Target calories per meal
            protein_target: Target protein per meal (grams)
        
        Returns:
            List of suggested meals
        """
        suggestions = []
        
        for meal_key, meal_data in self.MEALS.items():
            for portion, nutrition in meal_data["portions"].items():
                # Simple scoring based on proximity to targets
                calorie_diff = abs(nutrition.calories - calorie_target) / calorie_target
                protein_diff = abs(nutrition.protein - protein_target) / protein_target if protein_target > 0 else 0
                
                score = 1 - (0.7 * calorie_diff + 0.3 * protein_diff)
                
                if score > 0.6:  # Threshold for suggestion
                    suggestions.append({
                        "meal_key": meal_key,
                        "meal_name": meal_data["name"],
                        "portion_size": portion,
                        "nutrition": nutrition,
                        "score": round(score, 2),
                        "category": meal_data["category"]
                    })
        
        # Sort by score (highest first)
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions[:5]  # Return top 5 suggestions
    
    def get_daily_summary(self) -> Dict:
        """Get summary of all logged meals for the day"""
        if not self.logged_meals:
            return {
                "total_calories": 0,
                "total_protein": 0,
                "total_carbs": 0,
                "total_fat": 0,
                "meal_count": 0
            }
        
        total = NutritionalInfo(0, 0, 0, 0)
        for meal in self.logged_meals:
            total += meal["nutrition"]
        
        return {
            "total_calories": round(total.calories, 1),
            "total_protein": round(total.protein, 1),
            "total_carbs": round(total.carbs, 1),
            "total_fat": round(total.fat, 1),
            "meal_count": len(self.logged_meals)
        }
    
    def search_meals(self, query: str, category: Optional[str] = None) -> List[Dict]:
        """Search for meals by name or category"""
        results = []
        query = query.lower()
        
        for meal_key, meal_data in self.MEALS.items():
            if (query in meal_data["name"].lower() or query in meal_key) and \
               (category is None or meal_data.get("category") == category):
                
                results.append({
                    "key": meal_key,
                    "name": meal_data["name"],
                    "category": meal_data.get("category", "Unknown"),
                    "description": meal_data.get("description", ""),
                    "portions": list(meal_data["portions"].keys())
                })
        
        return results
    
    def export_logs(self, format: str = "json") -> Union[str, Dict]:
        """Export meal logs in specified format"""
        if format == "json":
            return {
                "meals": self.logged_meals,
                "summary": self.get_daily_summary(),
                "exported_at": self._get_timestamp()
            }
        else:
            # For text format
            summary = self.get_daily_summary()
            output = f"Daily Meal Log Summary\n"
            output += f"="*50 + "\n"
            output += f"Total Meals: {summary['meal_count']}\n"
            output += f"Total Calories: {summary['total_calories']} kcal\n"
            output += f"Total Protein: {summary['total_protein']}g\n"
            output += f"Total Carbs: {summary['total_carbs']}g\n"
            output += f"Total Fat: {summary['total_fat']}g\n"
            output += f"\nLogged Meals:\n"
            
            for i, meal in enumerate(self.logged_meals, 1):
                output += f"{i}. {meal['meal_name']} ({meal['portion_size']})\n"
                output += f"   Calories: {meal['nutrition'].calories} | "
                output += f"Protein: {meal['nutrition'].protein}g | "
                output += f"Carbs: {meal['nutrition'].carbs}g | "
                output += f"Fat: {meal['nutrition'].fat}g\n"
            
            return output
    
    def _get_timestamp(self) -> str:
        """Get current timestamp (simplified for demo)"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def print_nutrition_label(self, nutrition: NutritionalInfo, label: str = "Nutrition Info"):
        """Print nutrition information in a formatted way"""
        print(f"\n{label}")
        print("-" * 40)
        print(f"Calories: {nutrition.calories} kcal")
        print(f"Protein:  {nutrition.protein} g")
        print(f"Carbs:    {nutrition.carbs} g")
        print(f"Fat:      {nutrition.fat} g")
        print("-" * 40)


# Example usage and demonstration
def demonstrate_meal_logger():
    """Demonstrate the functionality of the Indian Meal Logger"""
    
    print("=" * 60)
    print("INDIAN FITNESS MEAL LOGGER")
    print("=" * 60)
    
    # Initialize the logger
    logger = IndianMealLogger()
    
    # Example 1: Quick logging of common meals
    print("\n1. QUICK MEAL LOGGING EXAMPLES:")
    print("-" * 40)
    
    # Log daal bhat with regular portion
    meal1 = logger.log_meal("daal_bhat", "Regular")
    print(f"\nLogged: {meal1['meal_name']} ({meal1['portion_size']})")
    logger.print_nutrition_label(meal1['nutrition'])
    
    # Log chole bhature with heavy portion
    meal2 = logger.log_meal("chole_bhature", "Heavy")
    print(f"\nLogged: {meal2['meal_name']} ({meal2['portion_size']})")
    logger.print_nutrition_label(meal2['nutrition'])
    
    # Example 2: Detailed component-based logging
    print("\n\n2. DETAILED COMPONENT LOGGING:")
    print("-" * 40)
    
    # Custom daal bhat with specific portions
    custom_components = [
        ("rice_basmati_cooked", 1.5),  # 1.5 cups of rice
        ("daal_toor", 2.0),            # 2 katoris of toor dal
        ("ghee", 2.0),                 # 2 tsp of ghee
        ("curd_cup", 0.5)              # half cup of curd
    ]
    
    meal3 = logger.log_meal("daal_bhat", "Custom", custom_components)
    print(f"\nLogged Custom: {meal3['meal_name']}")
    logger.print_nutrition_label(meal3['nutrition'], "Custom Meal Nutrition")
    
    # Example 3: Meal suggestions
    print("\n\n3. MEAL SUGGESTIONS (for ~400 calories, 15g protein):")
    print("-" * 40)
    
    suggestions = logger.suggest_meal(calorie_target=400, protein_target=15)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion['meal_name']} ({suggestion['portion_size']})")
        print(f"   Score: {suggestion['score']}")
        print(f"   Calories: {suggestion['nutrition'].calories}, Protein: {suggestion['nutrition'].protein}g")
    
    # Example 4: Search functionality
    print("\n\n4. SEARCH FOR MEALS:")
    print("-" * 40)
    
    search_results = logger.search_meals("biryani")
    print(f"Found {len(search_results)} biryani options:")
    for result in search_results:
        print(f"  - {result['name']} ({result['category']})")
    
    # Example 5: Daily summary
    print("\n\n5. DAILY SUMMARY:")
    print("-" * 40)
    
    summary = logger.get_daily_summary()
    print(f"Meals logged today: {summary['meal_count']}")
    print(f"Total calories: {summary['total_calories']} kcal")
    print(f"Total protein: {summary['total_protein']} g")
    
    # Example 6: Export logs
    print("\n\n6. EXPORT LOGS:")
    print("-" * 40)
    
    text_export = logger.export_logs(format="text")
    print(text_export)
    
    # Show total meals in database
    print(f"\n\nTotal meals in database: {len(logger.MEALS)}")
    print(f"Total components available: {len(logger.MEAL_COMPONENTS)}")
    
    return logger


if __name__ == "__main__":
    # Run the demonstration
    logger = demonstrate_meal_logger()
    
    # Additional utility: List all available meals
    print("\n" + "=" * 60)
    print("AVAILABLE MEALS IN DATABASE:")
    print("=" * 60)
    
    categories = {}
    for meal_key, meal_data in logger.MEALS.items():
        category = meal_data.get("category", "Uncategorized")
        if category not in categories:
            categories[category] = []
        categories[category].append(meal_data["name"])
    
    for category, meals in categories.items():
        print(f"\n{category} ({len(meals)} meals):")
        print("-" * 30)
        for meal in sorted(meals):
            print(f"  • {meal}")