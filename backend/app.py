from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date
import sys
import os
import re 
from dotenv import load_dotenv
import pandas as pd 
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.database import db, init_db, User, UserPlan, ChatMessage, MealLog, ProgressLog
from agent.fitness_agent import FitnessAgent
from agent.utils import EnsembleWeightPredictor, EnsembleAdherencePredictor, EnsembleMacroRecommender

# ── Gemini Setup ───────────────────────────────────────────────────────────────
from google import genai
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
print("✅ Gemini AI loaded")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{os.path.join(BASE_DIR, '..', 'instance', 'fitness_app.db')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

CORS(app)
init_db(app)
agent = FitnessAgent(models_path='models/trained_models')
print("Agent loaded successfully")


def get_user_by_id(user_id):
    return User.query.get(user_id)


def create_response(success=True, message=None, data=None, status_code=200):
    response = {'success': success, 'message': message, 'data': data}
    return jsonify(response), status_code


def build_gemini_system_prompt(user, plan):
    """Build a context-aware system prompt using user's actual plan."""
    profile = user.get_profile_dict()

    plan_overview = plan.get('plan_overview', {}) if plan else {}
    macros        = plan_overview.get('macros', {})
    targets       = plan.get('weekly_targets', {})

    return f"""You are FitAI, a friendly and knowledgeable personal fitness coach assistant.

USER PROFILE:
- Name: {user.username}
- Age: {profile.get('age')} | Gender: {profile.get('gender')}
- Weight: {profile.get('weight')} kg | Height: {profile.get('height')} cm
- Goal: {profile.get('goal', 'Not set').replace('_', ' ').title()}
- Activity Level: {profile.get('activity_level', 'moderate')}
- Gym Days: {profile.get('gym_days')} days/week

CURRENT AI-GENERATED PLAN (from Personalized Federated Learning):
- Daily Calories: {plan_overview.get('daily_calories', 'N/A')} kcal
- Protein: {macros.get('protein_g', 'N/A')}g | Carbs: {macros.get('carbs_g', 'N/A')}g | Fat: {macros.get('fat_g', 'N/A')}g
- Plan Difficulty: {plan_overview.get('difficulty', 'moderate')}
- Weekly Target: {targets.get('description', 'N/A')}
- Water Intake: {targets.get('water_intake', 'N/A')}L/day

PRIVACY NOTE: This user's plan was generated using Federated Learning — their data never left their device.

YOUR ROLE:
- Answer fitness, nutrition, and workout questions
- Give advice consistent with the user's current plan above
- Suggest Indian meals when giving food recommendations
- Be encouraging, specific, and concise
- If asked about medical conditions, recommend consulting a doctor
- Keep responses under 150 words unless the user asks for detail
"""

NUTRITION_DF = None
 
def load_nutrition_data():
    global NUTRITION_DF
    try:
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nutrition.csv'))
        NUTRITION_DF = df
        print(f"✅ Nutrition database loaded: {len(df)} foods")
    except Exception as e:
        print(f"⚠️  Nutrition data not loaded: {e}")
        NUTRITION_DF = None
 
def parse_numeric(value):
    """Extract numeric value from strings like '0.1g', '9.00 mg', '381'"""
    if pd.isna(value):
        return 0.0
    try:
        return float(re.sub(r'[^\d.]', '', str(value)))
    except:
        return 0.0
 
load_nutrition_data()

# ── AUTHENTICATION ─────────────────────────────────────────────────────────────
@app.route('/api/auth/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        required = ['username', 'email', 'password', 'age', 'gender', 'height',
                    'weight', 'goal', 'activity_level', 'gym_days']
        for field in required:
            if field not in data:
                return create_response(False, f'Missing field: {field}', status_code=400)

        if User.query.filter_by(username=data['username']).first():
            return create_response(False, 'Username already exists', status_code=400)
        if User.query.filter_by(email=data['email']).first():
            return create_response(False, 'Email already exists', status_code=400)

        new_user = User(
            username=data['username'],
            email=data['email'],
            password_hash=generate_password_hash(data['password'])
        )
        new_user.set_age(data['age'])
        new_user.set_gender(data['gender'])
        new_user.set_height(data['height'])
        new_user.set_weight(data['weight'])
        new_user.set_goal(data['goal'])
        new_user.set_activity_level(data['activity_level'])
        new_user.set_gym_days(data['gym_days'])

        db.session.add(new_user)
        db.session.commit()

        user_profile  = new_user.get_profile_dict()
        initial_plan  = agent.create_personalized_plan(user_profile)
        user_plan     = UserPlan(user_id=new_user.id)
        user_plan.set_plan(initial_plan)
        db.session.add(user_plan)
        db.session.commit()

        return create_response(True, 'User created successfully', {
            'user_id': new_user.id,
            'username': new_user.username,
            'plan_created': True
        })

    except Exception as e:
        db.session.rollback()
        return create_response(False, f'Error: {str(e)}', status_code=500)


@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.json
        user = User.query.filter_by(username=data['username']).first()
        if not user or not check_password_hash(user.password_hash, data['password']):
            return create_response(False, 'Invalid credentials', status_code=401)

        user.last_login = datetime.utcnow()
        db.session.commit()

        return create_response(True, 'Login successful', {
            'user_id': user.id,
            'username': user.username,
            'email': user.email
        })

    except Exception as e:
        return create_response(False, f'Error: {str(e)}', status_code=500)


# ── USER PROFILE & PLAN ────────────────────────────────────────────────────────
@app.route('/api/user/<int:user_id>/profile', methods=['GET'])
def get_user_profile(user_id):
    try:
        user = get_user_by_id(user_id)
        if not user:
            return create_response(False, 'User not found', status_code=404)

        profile = user.get_profile_dict()
        profile['username'] = user.username
        profile['email']    = user.email
        return create_response(True, 'Profile retrieved', profile)

    except Exception as e:
        return create_response(False, f'Error: {str(e)}', status_code=500)


@app.route('/api/user/<int:user_id>/plan', methods=['GET'])
def get_user_plan(user_id):
    try:
        user = get_user_by_id(user_id)
        if not user:
            return create_response(False, 'User not found', status_code=404)
        if not user.current_plan:
            return create_response(False, 'No plan found', status_code=404)

        return create_response(True, 'Plan retrieved', user.current_plan.get_plan())

    except Exception as e:
        return create_response(False, f'Error: {str(e)}', status_code=500)


# ── MEAL LOGGING ───────────────────────────────────────────────────────────────
@app.route('/api/user/<int:user_id>/meals', methods=['POST'])
def log_meal(user_id):
    try:
        user = get_user_by_id(user_id)
        if not user:
            return create_response(False, 'User not found', status_code=404)

        data     = request.json
        meal_log = MealLog(
            user_id   = user_id,
            meal_type = data.get('meal_type', 'meal'),
            date      = datetime.strptime(data['date'], '%Y-%m-%d').date() if 'date' in data else date.today(),
            calories  = data.get('calories', 0),
            protein   = data.get('protein', 0),
            carbs     = data.get('carbs', 0),
            fat       = data.get('fat', 0)
        )
        meal_log.set_meal_name(data.get('meal_name', 'Unknown'))
        db.session.add(meal_log)
        db.session.commit()

        return create_response(True, 'Meal logged successfully', meal_log.to_dict())

    except Exception as e:
        db.session.rollback()
        return create_response(False, f'Error: {str(e)}', status_code=500)


@app.route('/api/user/<int:user_id>/meals', methods=['GET'])
def get_meals(user_id):
    try:
        user = get_user_by_id(user_id)
        if not user:
            return create_response(False, 'User not found', status_code=404)

        date_str = request.args.get('date')
        if date_str:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            meals = MealLog.query.filter_by(user_id=user_id, date=target_date).all()
        else:
            meals = MealLog.query.filter_by(user_id=user_id).order_by(MealLog.date.desc()).limit(50).all()

        return create_response(True, 'Meals retrieved', [m.to_dict() for m in meals])

    except Exception as e:
        return create_response(False, f'Error: {str(e)}', status_code=500)


# ── PROGRESS TRACKING ──────────────────────────────────────────────────────────
@app.route('/api/user/<int:user_id>/progress', methods=['POST'])
def log_progress(user_id):
    try:
        user = get_user_by_id(user_id)
        if not user:
            return create_response(False, 'User not found', status_code=404)

        data     = request.json
        progress = ProgressLog(
            user_id             = user_id,
            week_number         = data.get('week_number', 1),
            date                = date.today(),
            workouts_completed  = data.get('workouts_completed', 0),
            avg_calories        = data.get('avg_calories', 0),
            adherence_score     = data.get('adherence_score', 0.0)
        )
        progress.set_weight(data['weight'])
        if 'notes' in data:
            progress.set_notes(data['notes'])

        user_profile = user.get_profile_dict()
        current_plan = user.current_plan.get_plan() if user.current_plan else None
        adaptation   = None

        if current_plan:
            progress_data = {
                'current_weight': data['weight'],
                'avg_calories':   data.get('avg_calories', current_plan['plan_overview']['daily_calories']),
                'avg_protein':    data.get('avg_protein', current_plan['plan_overview']['macros']['protein_g']),
                'workouts_done':  data.get('workouts_completed', 0)
            }
            adaptation = agent.adapt(user_profile, progress_data, current_plan)
            progress.predicted_weight = adaptation.get('predicted_weight', data['weight'])

            if adaptation['status'] != 'on_track':
                current_plan['plan_overview']['daily_calories'] = adaptation['new_calorie_target']
                user.current_plan.set_plan(current_plan)

        db.session.add(progress)
        db.session.commit()

        return create_response(True, 'Progress logged successfully', {
            'progress':   progress.to_dict(),
            'adaptation': adaptation
        })

    except Exception as e:
        db.session.rollback()
        return create_response(False, f'Error: {str(e)}', status_code=500)


@app.route('/api/user/<int:user_id>/progress', methods=['GET'])
def get_progress(user_id):
    try:
        user = get_user_by_id(user_id)
        if not user:
            return create_response(False, 'User not found', status_code=404)

        logs = ProgressLog.query.filter_by(user_id=user_id).order_by(ProgressLog.week_number).all()
        return create_response(True, 'Progress retrieved', [l.to_dict() for l in logs])

    except Exception as e:
        return create_response(False, f'Error: {str(e)}', status_code=500)


# ── CHAT WITH GEMINI ───────────────────────────────────────────────────────────
@app.route('/api/user/<int:user_id>/chat', methods=['POST'])
def chat(user_id):
    """
    Chat with Gemini AI fitness coach.
    Body: {message}
    """
    try:
        user = get_user_by_id(user_id)
        if not user:
            return create_response(False, 'User not found', status_code=404)

        data         = request.json
        user_message = data.get('message', '').strip()

        if not user_message:
            return create_response(False, 'Message cannot be empty', status_code=400)

        # Save user message to DB
        user_chat = ChatMessage(user_id=user_id, role='user')
        user_chat.set_message(user_message)
        db.session.add(user_chat)

        # Get user's current plan for context
        current_plan = user.current_plan.get_plan() if user.current_plan else {}

        # Build last 10 messages as conversation history
        recent_messages = ChatMessage.query.filter_by(user_id=user_id)\
            .order_by(ChatMessage.timestamp.desc()).limit(10).all()
        recent_messages = list(reversed(recent_messages))

        # Build Gemini conversation
        system_prompt = build_gemini_system_prompt(user, current_plan)

        # Format history for Gemini
        # Build Gemini conversation with history
        system_prompt = build_gemini_system_prompt(user, current_plan)

        # Format history as conversation turns
        contents = [{"role": "user", "parts": [{"text": system_prompt}]},
                    {"role": "model", "parts": [{"text": "Understood! I'm FitAI, your personal fitness coach. I have your profile and plan. How can I help?"}]}]

        # Add last 10 messages as history
        for msg in recent_messages:
            role = "user" if msg.role == "user" else "model"
            try:
                content = msg.get_message()
            except:
                content = ""
            if content:
                contents.append({"role": role, "parts": [{"text": content}]})

        # Add current message
        contents.append({"role": "user", "parts": [{"text": user_message}]})

        # Call Gemini with full history
        gemini_response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=contents
        )
        agent_response = gemini_response.text

        # Save agent response to DB
        agent_chat = ChatMessage(user_id=user_id, role='agent')
        agent_chat.set_message(agent_response)
        db.session.add(agent_chat)
        db.session.commit()

        return create_response(True, 'Message sent', {
            'user_message':   user_message,
            'agent_response': agent_response
        })

    except Exception as e:
        db.session.rollback()
        return create_response(False, f'Gemini Error: {str(e)}', status_code=500)


@app.route('/api/user/<int:user_id>/chat/history', methods=['GET'])
def get_chat_history(user_id):
    try:
        user = get_user_by_id(user_id)
        if not user:
            return create_response(False, 'User not found', status_code=404)

        messages  = ChatMessage.query.filter_by(user_id=user_id).order_by(ChatMessage.timestamp).all()
        chat_data = [msg.to_dict() for msg in messages]
        return create_response(True, 'Chat history retrieved', chat_data)

    except Exception as e:
        return create_response(False, f'Error: {str(e)}', status_code=500)


# ── HEALTH CHECK ───────────────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health_check():
    return create_response(True, 'Server is running', {
        'status':             'healthy',
        'agent_loaded':       agent is not None,
        'gemini_loaded':      gemini_model is not None,
        'database_connected': True
    })

@app.route('/api/nutrition/search', methods=['GET'])
def search_nutrition():
    """
    Search nutrition database
    Query params: ?q=chicken&limit=10
    """
    try:
        query = request.args.get('q', '').strip().lower()
        limit = int(request.args.get('limit', 10))
 
        if not query or len(query) < 2:
            return create_response(False, 'Query too short', status_code=400)
 
        if NUTRITION_DF is None:
            return create_response(False, 'Nutrition database not available', status_code=500)
 
        # Search by name
        mask = NUTRITION_DF['name'].str.lower().str.contains(query, na=False)
        results = NUTRITION_DF[mask].head(limit)
 
        foods = []
        for _, row in results.iterrows():
            foods.append({
                'name':     row['name'],
                'serving':  row.get('serving_size', '100g'),
                'calories': parse_numeric(row.get('calories', 0)),
                'protein':  parse_numeric(row.get('protein', 0)),
                'carbs':    parse_numeric(row.get('carbohydrate', 0)),
                'fat':      parse_numeric(row.get('fat', 0)),
                'fiber':    parse_numeric(row.get('fiber', 0)),
            })
 
        return create_response(True, f'Found {len(foods)} results', foods)
 
    except Exception as e:
        return create_response(False, f'Error: {str(e)}', status_code=500)
 
if __name__ == '__main__':
    print("🚀 STARTING FLASK SERVER")
    print("   http://localhost:5000")
    print("   Health: http://localhost:5000/api/health")
    app.run(debug=True, host='0.0.0.0', port=5000)