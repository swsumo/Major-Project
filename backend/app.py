from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import pandas as pd
import jwt
from functools import wraps
import os
import re
import secrets
import sys
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.database import db, init_db, User, UserPlan, ChatMessage, MealLog, ProgressLog, ExerciseLog, DATABASE_URI
from backend.agent.fitness_agent import FitnessAgent
from backend.agent.utils import (
    EnsembleWeightPredictor,
    EnsembleAdherencePredictor,
    EnsembleMacroRecommender
)

# ── Gemini Setup ───────────────────────────────────────────────────────────────
from google import genai
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
print(" Gemini AI loaded")

JWT_SECRET = os.getenv('JWT_SECRET', 'fitai-jwt-secret-change-in-production')
JWT_EXPIRY_HOURS = 24

password_reset_tokens = {}

app = Flask(__name__)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')


app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

CORS(app)
init_db(app)
#new added thing for render
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'login.html')

@app.route('/<path:filename>')
def serve_frontend(filename):
    return send_from_directory(FRONTEND_DIR, filename)
agent = FitnessAgent(models_path='models/trained_models')
print("Agent loaded successfully")


def get_user_by_id(user_id):
    return User.query.get(user_id)


def create_response(success=True, message=None, data=None, status_code=200):
    response = {'success': success, 'message': message, 'data': data}
    return jsonify(response), status_code


def generate_token(user_id: int, username: str) -> str:
    """Generate JWT token for user."""
    payload = {
        'user_id':  user_id,
        'username': username,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
        'iat':      datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')
 
 
def verify_token(token: str) -> dict:
    """Verify JWT token and return payload."""
    return jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
 
 
def token_required(f):
    """Decorator to protect routes with JWT."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
 
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
 
        if not token:
            return create_response(False, 'Authentication required', status_code=401)
 
        try:
            payload = verify_token(token)
            # Inject user_id from token into kwargs if route uses it
            request.current_user_id = payload['user_id']
        except jwt.ExpiredSignatureError:
            return create_response(False, 'Token expired — please login again', status_code=401)
        except jwt.InvalidTokenError:
            return create_response(False, 'Invalid token', status_code=401)
 
        return f(*args, **kwargs)
    return decorated
 
 
def token_matches_user(f):
    """Decorator — ensures token user_id matches the route user_id."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
 
        if not token:
            return create_response(False, 'Authentication required', status_code=401)
 
        try:
            payload  = verify_token(token)
            token_uid = payload['user_id']
            route_uid = kwargs.get('user_id')
 
            if route_uid and token_uid != route_uid:
                return create_response(False, 'Access denied', status_code=403)
 
            request.current_user_id = token_uid
        except jwt.ExpiredSignatureError:
            return create_response(False, 'Token expired — please login again', status_code=401)
        except jwt.InvalidTokenError:
            return create_response(False, 'Invalid token', status_code=401)
 
        return f(*args, **kwargs)
    return decorated

# ── REPLACE your existing build_gemini_system_prompt function with this ────────
 
def build_gemini_system_prompt(user, plan, user_id=None):
    """Build context-aware system prompt using user's plan + DB data."""
    profile      = user.get_profile_dict()
    plan_overview = plan.get('plan_overview', {}) if plan else {}
    macros        = plan_overview.get('macros', {})
    targets       = plan.get('weekly_targets', {})
 
    # ── Fetch recent meals from DB ─────────────────────────────────────────────
    meals_context = ""
    if user_id:
        try:
            recent_meals = MealLog.query.filter_by(user_id=user_id)\
                .order_by(MealLog.date.desc()).limit(10).all()
            if recent_meals:
                meals_lines = []
                for m in recent_meals:
                    meals_lines.append(
                        f"  - {m.date}: {m.get_meal_name()} ({m.meal_type}) — "
                        f"{m.calories}cal, P:{m.protein}g"
                    )
                meals_context = "\nRECENT MEALS (last 10 logged):\n" + "\n".join(meals_lines)
        except:
            pass
 
    # ── Fetch recent exercises from DB ─────────────────────────────────────────
    exercise_context = ""
    if user_id:
        try:
            recent_exercises = ExerciseLog.query.filter_by(user_id=user_id)\
                .order_by(ExerciseLog.date.desc()).limit(10).all()
            if recent_exercises:
                ex_lines = []
                for ex in recent_exercises:
                    ex_lines.append(
                        f"  - {ex.date}: {ex.exercise_name} ({ex.body_part}) — "
                        f"{ex.calories_burned}cal burned"
                        + (f", {ex.weight_kg}kg" if ex.weight_kg > 0 else "")
                    )
                exercise_context = "\nRECENT EXERCISES (last 10 logged):\n" + "\n".join(ex_lines)
        except:
            pass
 
    # ── Fetch progress logs from DB ────────────────────────────────────────────
    progress_context = ""
    if user_id:
        try:
            progress_logs = ProgressLog.query.filter_by(user_id=user_id)\
                .order_by(ProgressLog.week_number.desc()).limit(4).all()
            if progress_logs:
                pg_lines = []
                for p in progress_logs:
                    pg_lines.append(
                        f"  - Week {p.week_number}: {p.weight}kg "
                        f"(predicted: {p.predicted_weight}kg, "
                        f"adherence: {round(p.adherence_score*100) if p.adherence_score else 'N/A'}%)"
                    )
                progress_context = "\nRECENT PROGRESS (last 4 weeks):\n" + "\n".join(pg_lines)
        except:
            pass
 
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
{meals_context}
{exercise_context}
{progress_context}
 
YOUR ROLE:
- Answer fitness, nutrition, and workout questions
- Use the meal and exercise history above to give personalized advice
- If asked what the user ate on a specific date, check RECENT MEALS above
- If asked about their workout history, check RECENT EXERCISES above
- Suggest Indian meals when giving food recommendations
- Be encouraging, specific, and concise
- Keep responses under 150 words unless the user asks for detail
- Remove all markdown formatting like ** or ## from your responses
- If asked about medical conditions, recommend consulting a doctor
"""

NUTRITION_DF = None
 
def load_nutrition_data():
    global NUTRITION_DF
    try:
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'food.csv'))
        NUTRITION_DF = df
        print(f" Nutrition database loaded: {len(df)} foods")
    except Exception as e:
        print(f" Nutrition data not loaded: {e}")
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

        if User.query.filter_by(username=data['username']).first():
            return create_response(False, 'Username already exists', status_code=400)
        if User.query.filter_by(email=data['email']).first():
            return create_response(False, 'Email already exists', status_code=400)
        if not 16 <= int(data.get('age', 0)) <= 80:
            return create_response(False, 'Age must be between 16 and 80', status_code=400)
        if not 30 <= float(data.get('weight', 0)) <= 300:
            return create_response(False, 'Weight must be between 30 and 300 kg', status_code=400)

        required = ['username', 'email', 'password', 'age', 'gender', 'height',
                    'weight', 'goal', 'activity_level', 'gym_days']
        for field in required:
            if field not in data:
                return create_response(False, f'Missing field: {field}', status_code=400)

        
            
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

        token = generate_token(new_user.id, new_user.username)
        return create_response(True, 'User created successfully', {
            'user_id':      new_user.id,
            'username':     new_user.username,
            'plan_created': True,
            'token':        token
        })

    except Exception as e:
        db.session.rollback()
        return create_response(False, f'Error: {str(e)}', status_code=500)


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login and return JWT token."""
    try:
        data = request.json
        user = User.query.filter_by(username=data['username']).first()
 
        if not user or not check_password_hash(user.password_hash, data['password']):
            return create_response(False, 'Invalid credentials', status_code=401)
 
        user.last_login = datetime.utcnow()
        db.session.commit()
 
        token = generate_token(user.id, user.username)
 
        return create_response(True, 'Login successful', {
            'user_id':  user.id,
            'username': user.username,
            'email':    user.email,
            'token':    token
        })
 
    except Exception as e:
        return create_response(False, f'Error: {str(e)}', status_code=500)



@app.route('/api/auth/forgot-password', methods=['POST'])
def forgot_password():
    """Check if email exists. Body: {email}"""
    try:
        data  = request.json
        email = data.get('email', '').strip()
        if not email:
            return create_response(False, 'Email required', status_code=400)

        user = User.query.filter_by(email=email).first()
        if not user:
            return create_response(False, 'No account found with that email', status_code=404)

        # Generate a simple short-lived token (no email sent)
        token = secrets.token_urlsafe(32)
        password_reset_tokens[token] = {
            'user_id': user.id,
            'email':   email,
            'expires': datetime.utcnow() + timedelta(hours=1)
        }

        return create_response(True, 'Email verified', {'token': token})

    except Exception as e:
        return create_response(False, f'Error: {str(e)}', status_code=500)



@app.route('/api/auth/reset-password', methods=['POST'])
def reset_password():
    """Reset password with token. Body: {token, new_password}"""
    try:
        data         = request.json
        token        = data.get('token', '')
        new_password = data.get('new_password', '')
 
        if not token or not new_password:
            return create_response(False, 'Token and new password required', status_code=400)
 
        if len(new_password) < 6:
            return create_response(False, 'Password must be at least 6 characters', status_code=400)
 
        # Verify token
        token_data = password_reset_tokens.get(token)
        if not token_data:
            return create_response(False, 'Invalid or expired reset link', status_code=400)
 
        if datetime.utcnow() > token_data['expires']:
            del password_reset_tokens[token]
            return create_response(False, 'Reset link expired — request a new one', status_code=400)
 
        # Update password
        user = get_user_by_id(token_data['user_id'])
        if not user:
            return create_response(False, 'User not found', status_code=404)
 
        user.password_hash = generate_password_hash(new_password)
        db.session.commit()
 
        # Invalidate token
        del password_reset_tokens[token]
 
        return create_response(True, 'Password reset successfully — please login')
 
    except Exception as e:
        return create_response(False, f'Error: {str(e)}', status_code=500)



# ── USER PROFILE & PLAN ────────────────────────────────────────────────────────
@app.route('/api/user/<int:user_id>/profile', methods=['GET'])
@token_matches_user
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
@token_matches_user
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
@token_matches_user
def log_meal(user_id):
    try:
        data     = request.json
        
        user = get_user_by_id(user_id)
        if not user:
            return create_response(False, 'User not found', status_code=404)
        if data.get('calories', 0) < 0 or data.get('calories', 0) > 10000:
            return create_response(False, 'Invalid calories', status_code=400)
        if data.get('protein', 0) < 0 or data.get('protein', 0) > 500:
            return create_response(False, 'Invalid protein', status_code=400)

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
@token_matches_user
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
@token_matches_user
def log_progress(user_id):
    try:
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

        user = get_user_by_id(user_id)
        if not user:
            return create_response(False, 'User not found', status_code=404)
        if data.get('weight', 0) < 30 or data.get('weight', 0) > 300:
            return create_response(False, 'Invalid weight', status_code=400)
        if data.get('week_number', 0) < 1 or data.get('week_number', 0) > 100:
            return create_response(False, 'Invalid week number', status_code=400)


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
@token_matches_user
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
@token_matches_user
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

        # ── RATE LIMIT: 10 user messages per day ─────────────────────────────
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_message_count = ChatMessage.query.filter(
            ChatMessage.user_id == user_id,
            ChatMessage.role == 'user',
            ChatMessage.timestamp >= today_start
        ).count()

        if today_message_count >= 10:
            return create_response(False, 'Daily limit reached. You can send up to 10 messages per day.', status_code=429)
        # ─────────────────────────────────────────────────────────────────────

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

        # Build system prompt with DB context
        system_prompt = build_gemini_system_prompt(user, current_plan, user_id=user_id)

        # Format history as conversation turns
        contents = [
            {"role": "user",  "parts": [{"text": system_prompt}]},
            {"role": "model", "parts": [{"text": "Understood! I'm FitAI, your personal fitness coach. I have your profile, meal history, and exercise logs. How can I help?"}]}
        ]

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

        # Call Gemini with full history + DB context
        gemini_response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=contents
        )
        agent_response = gemini_response.text

        # Clean markdown formatting
        import re
        agent_response = re.sub(r'\*\*(.*?)\*\*', r'\1', agent_response)
        agent_response = re.sub(r'\*(.*?)\*',     r'\1', agent_response)
        agent_response = re.sub(r'#{1,6}\s',      '',    agent_response)

        # Save agent response
        agent_chat = ChatMessage(user_id=user_id, role='agent')
        agent_chat.set_message(agent_response)
        db.session.add(agent_chat)
        db.session.commit()

        return create_response(True, 'Message sent', {
            'user_message':   user_message,
            'agent_response': agent_response,
            'messages_remaining': max(0, 10 - (today_message_count + 1))
        })

    except Exception as e:
        db.session.rollback()
        return create_response(False, f'Gemini Error: {str(e)}', status_code=500)



@app.route('/api/user/<int:user_id>/chat/history', methods=['GET'])
@token_matches_user
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

# ── EXERCISE LOGGING ROUTES ───────────────────────────────────────────────────
@app.route('/api/user/<int:user_id>/exercise', methods=['POST'])
@token_matches_user
def log_exercise(user_id):
    """
    Log an exercise
    Body: {exercise_name, body_part, weight_kg, sets_reps, workout_duration, calories_burned, date}
    """
    try:
        data = request.json
        user = get_user_by_id(user_id)
        if not user:
            return create_response(False, 'User not found', status_code=404)
        if data.get('calories_burned', 0) < 0 or data.get('calories_burned', 0) > 5000:
            return create_response(False, 'Invalid calories burned', status_code=400)

        exercise = ExerciseLog(
            user_id          = user_id,
            date             = datetime.strptime(data['date'], '%Y-%m-%d').date() if 'date' in data else date.today(),
            exercise_name    = data.get('exercise_name', 'Unknown'),
            body_part        = data.get('body_part', 'Full Body'),
            weight_kg        = data.get('weight_kg', 0),
            sets_reps        = data.get('sets_reps', ''),
            workout_duration = data.get('workout_duration', 0),
            calories_burned  = data.get('calories_burned', 0)
        )
        db.session.add(exercise)
        db.session.commit()
        return create_response(True, 'Exercise logged successfully', exercise.to_dict())

    except Exception as e:
        db.session.rollback()
        return create_response(False, f'Error: {str(e)}', status_code=500)


@app.route('/api/user/<int:user_id>/exercise', methods=['GET'])
@token_matches_user
def get_exercises(user_id):
    """Get exercise logs — optional ?date=YYYY-MM-DD filter"""
    try:
        user = get_user_by_id(user_id)
        if not user:
            return create_response(False, 'User not found', status_code=404)

        date_str = request.args.get('date')
        if date_str:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            logs = ExerciseLog.query.filter_by(user_id=user_id, date=target_date).all()
        else:
            logs = ExerciseLog.query.filter_by(user_id=user_id)\
                .order_by(ExerciseLog.date.desc()).limit(50).all()

        # Compute totals
        total_calories = sum(l.calories_burned for l in logs)
        total_duration = sum(l.workout_duration for l in logs)
        body_parts     = list(set(l.body_part for l in logs if l.body_part))

        return create_response(True, 'Exercises retrieved', {
            'exercises':      [l.to_dict() for l in logs],
            'total_calories': total_calories,
            'total_duration': total_duration,
            'body_parts':     body_parts
        })

    except Exception as e:
        return create_response(False, f'Error: {str(e)}', status_code=500)


@app.route('/api/user/<int:user_id>/exercise/summary', methods=['GET'])
@token_matches_user
def get_exercise_summary(user_id):
    """Get weekly exercise summary — body parts trained this week"""
    try:
        from datetime import timedelta
        user = get_user_by_id(user_id)
        if not user:
            return create_response(False, 'User not found', status_code=404)

        # Last 7 days
        week_ago = date.today() - timedelta(days=7)
        logs = ExerciseLog.query.filter(
            ExerciseLog.user_id == user_id,
            ExerciseLog.date >= week_ago
        ).all()

        # Body part frequency
        body_part_counts = {}
        for log in logs:
            bp = log.body_part or 'Full Body'
            body_part_counts[bp] = body_part_counts.get(bp, 0) + 1

        total_calories = sum(l.calories_burned for l in logs)
        total_duration = sum(l.workout_duration for l in logs)
        total_sessions = len(set(l.date for l in logs))

        return create_response(True, 'Weekly summary retrieved', {
            'body_part_counts': body_part_counts,
            'total_calories':   total_calories,
            'total_duration':   total_duration,
            'total_sessions':   total_sessions,
            'exercises':        [l.to_dict() for l in logs]
        })

    except Exception as e:
        return create_response(False, f'Error: {str(e)}', status_code=500)

# ── HEALTH CHECK ───────────────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health_check():
    return create_response(True, 'Server is running', {
        'status':             'healthy',
        'agent_loaded':       agent is not None,
        'gemini_loaded':      gemini_client is not None,
        'database_connected': True
    })

@app.route('/api/nutrition/search', methods=['GET'])
@token_matches_user
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
                'name':       row['name'],
                'serving':    row.get('serving_size', '100g'),
                'calories':   parse_numeric(row.get('calories', 0)),
                'protein':    parse_numeric(row.get('protein', 0)),
                'carbs':      parse_numeric(row.get('carbohydrate', 0)),
                'fat':        parse_numeric(row.get('fat', 0)),
                'fiber':      parse_numeric(row.get('fiber', 0)),
            })
 
        return create_response(True, f'Found {len(foods)} results', foods)
 
    except Exception as e:
        return create_response(False, f'Error: {str(e)}', status_code=500) 


# ── UPDATE PROFILE + REGENERATE PLAN ─────────────────────────────────────────
@app.route('/api/user/<int:user_id>/profile', methods=['PUT'])
@token_matches_user
def update_profile(user_id):
    """
    Update user profile and regenerate FL plan.
    Body: {age, gender, height, weight, goal, activity_level, gym_days}
    """
    try:
        user = get_user_by_id(user_id)
        if not user:
            return create_response(False, 'User not found', status_code=404)

        data = request.json

        # Update only fields that are provided
        if 'age' in data:
            user.set_age(data['age'])
        if 'gender' in data:
            user.set_gender(data['gender'])
        if 'height' in data:
            user.set_height(data['height'])
        if 'weight' in data:
            user.set_weight(data['weight'])
        if 'goal' in data:
            user.set_goal(data['goal'])
        if 'activity_level' in data:
            user.set_activity_level(data['activity_level'])
        if 'gym_days' in data:
            user.set_gym_days(data['gym_days'])

        db.session.commit()

        # Regenerate FL plan with updated profile
        user_profile  = user.get_profile_dict()
        new_plan      = agent.create_personalized_plan(user_profile)

        # Update plan in database
        if user.current_plan:
            user.current_plan.set_plan(new_plan)
        else:
            user_plan = UserPlan(user_id=user.id)
            user_plan.set_plan(new_plan)
            db.session.add(user_plan)

        db.session.commit()

        return create_response(True, 'Profile updated and plan regenerated', {
            'profile': user.get_profile_dict(),
            'plan':    new_plan
        })

    except Exception as e:
        db.session.rollback()
        return create_response(False, f'Error: {str(e)}', status_code=500)

# Health Check Route
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "message": "Flask server is running "
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"STARTING SERVER ON PORT {port}")
    app.run(host='0.0.0.0', port=port)

