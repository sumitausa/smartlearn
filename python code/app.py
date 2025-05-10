# SmartLearn Quiz App - app.py
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, flash
from datetime import datetime, timedelta
import numpy as np
import joblib
import os
from quiz_generator import generate_quiz
from urllib.parse import quote
import csv
import pandas as pd
from openai import OpenAI

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Session configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Session lasts 7 days
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True

# Database setup
from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///quiz.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    quiz_attempts = db.relationship('QuizAttempt', backref='user', lazy=True)

class QuizAttempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    score = db.Column(db.Float, nullable=False)
    topic = db.Column(db.String(80), nullable=False)
    level = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

def init_db():
    with app.app_context():
        db.drop_all()
        db.create_all()
        print("Database tables created successfully!")

# Initialize database tables if they don't exist
with app.app_context():
    db.create_all()
    print("Database tables initialized!")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    clf = joblib.load("user_level_classifier.joblib")
    scaler = joblib.load("user_level_scaler.joblib")
    print("Loaded ML model and scaler.")
except:
    clf = None
    scaler = None
    print("Warning: ML model or scaler not found.")

NUM_QUESTIONS = 10

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        if not username:
            flash('Username is required')
            return redirect(url_for('login'))
            
        # Simple login - just set the username in session
        user = User.query.filter_by(username=username).first()
        if not user:
            user = User(username=username)
            db.session.add(user)
            db.session.commit()
            print(f"Created new user: {user.username} with id: {user.id}")
        else:
            print(f"Found existing user: {user.username} with id: {user.id}")
        
        # Set session data
        session.clear()  # Clear any existing session data
        session['username'] = username
        session['user_id'] = user.id
        session.permanent = True  # Make session permanent
        print(f"Set session for user: {username} (ID: {user.id})")
        
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/progress')
def view_progress():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get user's quiz attempts
    user = User.query.get(session['user_id'])
    if not user:
        return redirect(url_for('login'))
    
    # Get quiz attempts and scores
    attempts = QuizAttempt.query.filter_by(user_id=user.id).order_by(QuizAttempt.timestamp.desc()).all()
    
    # Calculate statistics
    scores = [attempt.score for attempt in attempts]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # Set default values
    total_quizzes = len(attempts)
    best_score = max(scores) if scores else 0
    recent_topics = [attempt.topic for attempt in attempts[:5]]
    user_level = "Not enough data"
    
    # Calculate level based on recent performance if there are attempts
    if attempts:
        recent_scores = scores[:5]  # Use last 5 attempts
        avg_recent_score = sum(recent_scores) / len(recent_scores)
        
        if avg_recent_score >= 80:
            user_level = "Advanced"
        elif avg_recent_score >= 60:
            user_level = "Intermediate"
        else:
            predicted_level = "Beginner"
    else:
        predicted_level = 'Not enough data'
    
    # Ensure all template variables are defined
    template_data = {
        'username': user.username,
        'user_level': user_level,
        'total_quizzes': total_quizzes,
        'avg_score': avg_score,
        'best_score': best_score,
        'recent_topics': recent_topics,
        'attempts': attempts
    }
    
    print("Rendering progress template with data:", template_data)
    return render_template('progress.html', **template_data)

def parse_quiz_content(content):
    """Parse quiz content from the API response"""
    if isinstance(content, dict) and 'questions' in content:
        return content['questions']
    return None

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('quiz.html')

@app.route('/generate', methods=['POST'])
def generate_quiz_route():
    print("\n=== Starting Quiz Generation ===\n")
    print(f"Current session data: {session}")
    
    if 'user_id' not in session:
        print("No user_id in session, redirecting to login")
        return redirect(url_for('login'))
        
    topic = request.form['topic']
    print(f"Generating quiz for topic: {topic}")
    
    content = generate_quiz(topic, NUM_QUESTIONS)
    questions = parse_quiz_content(content)
    print(f"Generated questions: {questions}")
    
    if not questions or isinstance(content, str) and content.startswith("Error"):
        print("Quiz generation failed")
        return render_template('quiz.html', questions=None, topic=topic, error="Quiz generation failed.")
    
    # Store questions in session
    try:
        session['questions'] = questions
        session['topic'] = topic
        session.permanent = True
        session.modified = True
        print(f"Stored in session - Questions count: {len(session['questions'])}")
        print(f"Updated session data: {session}")
    except Exception as e:
        print(f"Error storing in session: {str(e)}")
        return render_template('quiz.html', questions=None, topic=topic, error="Session storage failed.")
    
    print("\n=== Quiz Generation Complete ===\n")
    return render_template('quiz.html', questions=questions, topic=topic)

@app.route('/submit', methods=['POST'])
def submit_quiz():
    print("\n=== Starting Quiz Submission ===\n")
    print(f"Current session data: {session}")
    print(f"Form data: {request.form}")
    
    if 'user_id' not in session:
        print("No user_id in session")
        return redirect(url_for('login'))
    
    topic = request.form.get('topic', '')
    score, total = 0, 0
    results = []
    
    # Get user
    user = User.query.get(session['user_id'])
    if not user:
        print(f"User not found for id: {session['user_id']}")
        session.clear()
        return redirect(url_for('login'))
    
    print(f"Processing quiz for user: {user.username} (ID: {user.id})")

    # Get stored questions from session
    stored_questions = session.get('questions', [])
    print(f"Retrieved from session - Questions: {stored_questions}")
    
    if not stored_questions:
        print("No questions found in session")
        return redirect(url_for('index'))
    
    print(f"Processing {len(stored_questions)} questions from session")
    
    # Process quiz answers
    total = len(stored_questions)  # Total is number of questions
    for i, question in enumerate(stored_questions):
        print(f"\nProcessing Question {i + 1}:")
        print(f"Question data: {question}")
        
        user_answer = request.form.get(f'answer_{i}')
        print(f"User answer from form: {user_answer}")
        
        correct_answer = question.get('correct')
        print(f"Correct answer from question: {correct_answer}")
        
        # Calculate score and store result
        is_correct = user_answer == correct_answer if user_answer is not None else False
        if is_correct:
            score += 1
        
        # Store result
        result = {
            'question': question['question'],
            'user_answer': user_answer if user_answer is not None else 'Not answered',
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'explanation': question.get('explanation', '')
        }
        results.append(result)
        print(f"Added result: {result}")
    
    print(f"\nFinal Score: {score}/{total}")
    
    # Clear questions from session after processing
    try:
        session.pop('questions', None)
        session.modified = True
        print("Cleared questions from session")
        print(f"Updated session data: {session}")
    except Exception as e:
        print(f"Error clearing session: {str(e)}")

    percentage = (score / total * 100) if total > 0 else 0
    # Calculate final score percentage
    final_score = round((score / total * 100), 2) if total > 0 else 0
    print(f"Quiz score: {score}/{total} = {final_score}%")
    
    # Get recent quiz attempts to determine level
    attempts = QuizAttempt.query.filter_by(user_id=user.id).order_by(QuizAttempt.timestamp.desc()).all()
    
    # Calculate user level based on recent performance (last 5 attempts)
    if attempts:
        recent_scores = [attempt.score for attempt in attempts[:5]]  # Last 5 attempts
        avg_recent_score = sum(recent_scores) / len(recent_scores)
        
        if avg_recent_score >= 80:
            user_level = "Advanced"
        elif avg_recent_score >= 60:
            user_level = "Intermediate"
        else:
            user_level = "Beginner"
    else:
        user_level = "Beginner"  # First quiz attempt
    
    print(f"Assigned level based on recent performance: {user_level}")
    
    # Track missed concepts for recommendations
    missed_concepts = session.get('missed_concepts', {})
    next_topic = max(missed_concepts, key=missed_concepts.get) if missed_concepts else topic
    
    search_query = f"{next_topic} tutorial for {user_level.lower()}"
    youtube_url = f"https://www.youtube.com/results?search_query={quote(search_query)}"
    
    try:
        # Save quiz attempt
        quiz_attempt = QuizAttempt(
            user_id=user.id,
            topic=topic,
            score=final_score,
            level=user_level,
            timestamp=datetime.utcnow()
        )
        db.session.add(quiz_attempt)
        db.session.commit()
        print(f"Saved quiz attempt for user {user.username}: {final_score}% ({user_level})")
    except Exception as e:
        print(f"Error saving quiz attempt: {str(e)}")
        db.session.rollback()
    
    # Generate YouTube recommendations based on performance
    search_query = f"{topic} tutorial for {user_level.lower()} level"
    youtube_url = f"https://www.youtube.com/results?search_query={quote(search_query)}"
    
    return render_template('results.html',
                           topic=topic,
                           score=score,
                           total=total,
                           accuracy=final_score,
                           results=results,
                           youtube_url=youtube_url,
                           user_level=user_level)

@app.route('/progress')
def progress():
    print("\n=== Accessing Progress Page ===")
    if 'user_id' not in session:
        print("No user_id in session")
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        print(f"User not found for id: {session['user_id']}")
        session.clear()
        return redirect(url_for('login'))
    
    print(f"Getting progress for user: {user.username} (ID: {user.id})")
    
    # Initialize default values
    total_quizzes = 0
    avg_score = 0
    best_score = 0
    recent_topics = []
    user_level = "Not enough data"
    
    # Get quiz attempts and scores
    attempts = QuizAttempt.query.filter_by(user_id=user.id).order_by(QuizAttempt.timestamp.desc()).all()
    
    # Calculate statistics
    scores = [attempt.score for attempt in attempts]
    avg_score = round(sum(scores) / len(scores), 2) if scores else 0
    
    # Set default values
    total_quizzes = len(attempts)
    best_score = round(max(scores), 2) if scores else 0
    recent_topics = [attempt.topic for attempt in attempts[:5]]
    user_level = "Not enough data"
    
    # Calculate level based on recent performance if there are attempts
    if attempts:
        recent_scores = scores[:5]  # Use last 5 attempts
        avg_recent_score = sum(recent_scores) / len(recent_scores)
        
        if avg_recent_score >= 80:
            user_level = "Advanced"
        elif avg_recent_score >= 60:
            user_level = "Intermediate"
        else:
            user_level = "Beginner"
            
        print(f"Statistics calculated:")
        print(f"- Total quizzes: {total_quizzes}")
        print(f"- Average score: {avg_score}")
        print(f"- Best score: {best_score}")
        print(f"- User level: {user_level}")
        print(f"- Recent topics: {recent_topics}")
    else:
        print("No quiz attempts found - using default values")
    
    # Ensure all template variables are defined
    template_data = {
        'username': user.username,
        'user_level': user_level,
        'total_quizzes': total_quizzes,
        'avg_score': avg_score,
        'best_score': best_score,
        'recent_topics': recent_topics,
        'attempts': attempts
    }
    
    print("Rendering progress template with data:", template_data)
    return render_template('progress.html', **template_data)

@app.route('/thank_you', methods=['POST'])
def submit_feedback():
    helpful = request.form.get('helpful', 'No response')
    comments = request.form.get('comments', '').strip() or "No comment"
    feedback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feedback_data')
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_file = os.path.join(feedback_dir, 'feedback.csv')
    headers = ['Helpful', 'Comments']
    write_header = not os.path.exists(feedback_file)
    with open(feedback_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow([helpful, comments])
    return render_template('thank_you.html')

@app.route('/view_feedback')
def view_feedback():
    try:
        feedback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feedback_data')
        feedback_file = os.path.join(feedback_dir, 'feedback.csv')
        print("📂 Looking for feedback at:", feedback_file)

        if not os.path.exists(feedback_file):
            return render_template('view_feedback.html', feedback=[], summary="No feedback submitted yet.")

        df = pd.read_csv(feedback_file)
        if df.empty or "Helpful" not in df.columns or "Comments" not in df.columns:
            return render_template('view_feedback.html', feedback=[], summary="Invalid feedback format.")

        feedback_data = df.dropna(how='all').to_dict(orient='records')
        comments = df["Comments"].dropna().tolist()

        summary = None
        if comments:
            print("💬 Sending comments to OpenAI for summary...")
            prompt = (
                "Summarize the following user feedback into:\n"
                "1. Common praise\n"
                "2. Frequent issues\n"
                "3. Suggestions for improvement\n"
                "4. Recommended next features\n\n"
                "Feedback:\n" + "\n".join(comments)
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            summary = response.choices[0].message.content.strip()

        return render_template('view_feedback.html', feedback=feedback_data, summary=summary)

    except Exception as e:
        print("❌ Error loading feedback:", str(e))
        return render_template('view_feedback.html', feedback=[], summary=f"Error reading feedback: {str(e)}")

if __name__ == '__main__':
    app.run(debug=False, port=5006)
