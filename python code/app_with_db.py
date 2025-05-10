from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from models import db, User, QuizAttempt, ConceptProgress
import os
from quiz_generator import generate_quiz
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Get the absolute path to the database file
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'quiz_app.db')

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db.init_app(app)

# Create tables
with app.app_context():
    db.create_all()
    print(f'Database initialized at: {db_path}')

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('quiz'))
    return redirect(url_for('register'))

@app.route('/quiz')
def quiz():
    if 'user_id' not in session:
        return redirect(url_for('register'))
    user = User.query.get(session['user_id'])
    if not user:
        session.pop('user_id', None)
        return redirect(url_for('register'))
    level = get_user_level(user)
    topic = request.args.get('topic') or get_next_topic(user) or 'Python'
    quiz_data = generate_quiz(topic, level)
    return render_template('quiz.html', questions=quiz_data['questions'], topic=topic)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        print(f'Registration attempt for username: {username}')
        
        if not username:
            print('Error: Username is required')
            return render_template('register.html', error='Username is required')
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            print(f'Error: Username {username} already exists')
            return render_template('register.html', error='Username already exists')
            
        try:
            print(f'Creating new user: {username}')
            user = User(username=username)
            db.session.add(user)
            db.session.commit()
            print(f'User created with ID: {user.id}')
            
            session['user_id'] = user.id
            print(f'Session user_id set to: {session.get("user_id")}')
            
            return redirect(url_for('quiz'))
        except Exception as e:
            db.session.rollback()
            print(f'Error during registration: {str(e)}')
            return render_template('register.html', error=f'Registration failed: {str(e)}')
            
    return render_template('register.html')

@app.route('/generate', methods=['POST'])
def generate():
    if 'user_id' not in session:
        return redirect(url_for('register'))
    topic = request.form.get('topic')
    return redirect(url_for('quiz', topic=topic))

@app.route('/submit', methods=['POST'])
def submit_quiz():
    if 'user_id' not in session:
        return redirect(url_for('register'))
    
    user = User.query.get(session['user_id'])
    topic = request.form.get('topic', '')
    score = 0
    total = 0
    hint_usage = 0
    retries = 0
    
    # Calculate quiz results
    for key in request.form:
        if key.startswith('question_'):
            # Extract question index from the key (e.g., 'question_1' -> '1')
            idx = key.split('_')[1]
            correct = request.form.get(f'correct_answer_{idx}')
            answer = request.form.get(f'answer_{idx}')
            if answer == correct:
                score += 1
            total += 1

    # Save quiz attempt
    percentage = (score / total * 100) if total > 0 else 0
    quiz_attempt = QuizAttempt(
        user_id=user.id,
        topic=topic,
        score=percentage,
        hint_usage=hint_usage,
        retries=retries,
        difficulty_level=get_user_level(user)
    )
    db.session.add(quiz_attempt)
    db.session.commit()

    # Calculate percentage and get recommendations
    score_percentage = (score / total * 100) if total > 0 else 0
    next_topic = get_next_topic(user)
    
    # Generate YouTube URL for recommendations
    if not topic:  # Fallback to Python if no topic is provided
        topic = 'Python'
    search_query = f"{topic}+tutorial+for+beginners" if score_percentage < 70 else f"{topic}+advanced+tutorial"
    youtube_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
    
    # Debug logging
    print(f"Debug - Topic: {topic}")
    print(f"Debug - Score: {score}/{total} ({score_percentage}%)")
    print(f"Debug - YouTube URL: {youtube_url}")
    
    # Create results list for individual questions
    results = []
    for key in request.form:
        if key.startswith('question_'):
            idx = key.split('_')[1]
            question = request.form.get(f'question_{idx}')
            user_answer = request.form.get(f'answer_{idx}')
            correct_answer = request.form.get(f'correct_answer_{idx}')
            options = []
            for label in ['A', 'B', 'C', 'D']:
                option_text = request.form.get(f'option_{idx}_{label}')
                if option_text:
                    options.append({'label': label, 'text': option_text})
            
            explanation = request.form.get(f'explanation_{idx}')
            results.append({
                'question': question,
                'user_answer': user_answer,
                'correct_answer': correct_answer,
                'is_correct': user_answer == correct_answer,
                'options': options,
                'explanation': explanation
            })
    
    return render_template('results.html',
                         score=score,
                         total_questions=total,
                         score_percentage=score_percentage,
                         next_topic=next_topic,
                         topic=topic,
                         youtube_url=youtube_url,
                         results=results)

def get_user_level(user, specific_topic=None):
    """Determine user level based on recent performance.
    If specific_topic is provided, only consider attempts for that topic.
    Otherwise, use the overall performance across topics."""
    query = QuizAttempt.query.filter_by(user_id=user.id)
    
    if specific_topic:
        query = query.filter_by(topic=specific_topic)
    
    # Get recent attempts
    recent_attempts = query.order_by(QuizAttempt.timestamp.desc()).limit(5).all()
    
    if not recent_attempts:
        return 'Beginner'
    
    # Calculate average score
    avg_score = sum(attempt.score for attempt in recent_attempts) / len(recent_attempts)
    
    # Get topic-specific levels if no specific topic was requested
    if not specific_topic:
        # Group attempts by topic and get level for each
        topic_attempts = {}
        all_attempts = QuizAttempt.query.filter_by(user_id=user.id).all()
        
        for attempt in all_attempts:
            if attempt.topic not in topic_attempts:
                topic_attempts[attempt.topic] = []
            topic_attempts[attempt.topic].append(attempt.score)
        
        # Calculate level for each topic
        topic_levels = {}
        for topic, scores in topic_attempts.items():
            topic_avg = sum(scores) / len(scores)
            if topic_avg >= 80:
                topic_levels[topic] = 'Advanced'
            elif topic_avg >= 60:
                topic_levels[topic] = 'Intermediate'
            else:
                topic_levels[topic] = 'Beginner'
        
        # Store topic levels in session for display
        from flask import session
        session['topic_levels'] = topic_levels
    
    # Return overall level based on recent performance
    if avg_score >= 80:
        return 'Advanced'
    elif avg_score >= 60:
        return 'Intermediate'
    return 'Beginner'

def get_next_topic(user):
    """Recommend next topic based on most problematic concepts"""
    worst_concept = ConceptProgress.query\
        .filter_by(user_id=user.id)\
        .order_by(ConceptProgress.missed_count.desc())\
        .first()
    
    return worst_concept.concept if worst_concept else None

@app.route('/progress')
def view_progress():
    if 'user_id' not in session:
        return redirect(url_for('register'))
    
    user = User.query.get(session['user_id'])
    if not user:
        return redirect(url_for('register'))
    
    # Get quiz attempts (excluding ones with no score)
    quiz_attempts = QuizAttempt.query.filter_by(user_id=user.id)\
        .filter(QuizAttempt.score > 0)\
        .order_by(QuizAttempt.timestamp.desc())\
        .limit(10).all()
    
    # Get concept progress
    concept_progress = ConceptProgress.query.filter_by(user_id=user.id)\
        .order_by(ConceptProgress.missed_count.desc())\
        .limit(5).all()
    
    # Calculate overall stats
    total_quizzes = QuizAttempt.query.filter_by(user_id=user.id).count()
    avg_score = db.session.query(db.func.avg(QuizAttempt.score))\
        .filter_by(user_id=user.id).scalar() or 0
    
    return render_template('progress.html',
                          username=user.username,
                          level=get_user_level(user),
                          quiz_attempts=quiz_attempts,
                          concept_progress=concept_progress,
                          total_quizzes=total_quizzes,
                          avg_score=avg_score)

if __name__ == '__main__':
    app.run(debug=True, port=5012)
