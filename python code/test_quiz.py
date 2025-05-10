import unittest
from app import app, db, User, QuizAttempt
from datetime import datetime
from bs4 import BeautifulSoup
import json
import requests

class TestQuiz(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app_context = app.app_context()
        self.app_context.push()
        self.client = app.test_client()
        db.create_all()
        
        # Create or get test user
        self.test_user = User.query.filter_by(username='test_user').first()
        if not self.test_user:
            self.test_user = User(username='test_user')
            db.session.add(self.test_user)
            db.session.commit()
    
    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
    
    def test_quiz_submission(self):
        print("\n=== Starting Quiz Test ===\n")
        
        # Base URL for the application
        base_url = "http://127.0.0.1:5001"
        
        # Start a session to maintain cookies
        s = requests.Session()
        
        # Step 1: Login
        print("1. Logging in...")
        login_data = {"username": "test_user"}
        response = s.post(f"{base_url}/login", data=login_data)
        print(f"Login status: {response.status_code}")
        
        # Step 2: Generate a quiz
        print("\n2. Generating quiz...")
        quiz_data = {"topic": "Python basics"}
        response = s.post(f"{base_url}/generate", data=quiz_data)
        print(f"Quiz generation status: {response.status_code}")
        
        # Extract the form structure from the response HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all radio inputs in the form
        form = soup.find('form', {'action': '/submit'})
        if not form:
            print("Error: Could not find quiz form!")
            return
        
        # Collect all question data
        questions = []
        current_question = None
        
        for element in form.find_all(['p', 'input']):
            if element.name == 'p' and element.find('strong'):  # Question text
                if current_question:
                    questions.append(current_question)
                current_question = {'options': []}
                current_question['text'] = element.get_text()
            elif element.name == 'input' and element.get('type') == 'radio':
                if current_question:
                    current_question['options'].append({
                        'name': element.get('name'),
                        'value': element.get('value')
                    })
        
        if current_question:
            questions.append(current_question)
        
        print(f"\nFound {len(questions)} questions")
        for i, q in enumerate(questions):
            print(f"\nQuestion {i+1}:")
            print(f"Text: {q['text']}")
            print(f"Options: {len(q['options'])} radio buttons")
            for opt in q['options']:
                print(f"  - {opt['name']}: {opt['value']}")
        
        # Step 3: Submit answers for all questions
        print("\n3. Submitting quiz...")
        submit_data = {'topic': 'Python basics'}
        
        # Add an answer for each question (selecting option 'A' for all)
        for q in questions:
            if q['options']:
                submit_data[q['options'][0]['name']] = 'A'
        
        print("\nSubmitting form data:")
        print(json.dumps(submit_data, indent=2))
        
        response = s.post(f"{base_url}/submit", data=submit_data)
        print(f"\nSubmission status: {response.status_code}")
        
        # Check the response content
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='card-body')
        
        print("\n=== Quiz Results ===")
        for i, result in enumerate(results):
            answer = result.find('p', class_='text-primary')
            if answer:
                print(f"Question {i+1} Answer: {answer.get_text()}")
        
        print("\n=== Test Complete ===")

    def test_quiz_submission_with_unanswered(self):
        """Test that quiz submission works correctly with unanswered questions"""
        with self.client.session_transaction() as session:
            session['user_id'] = self.test_user.id
            session['username'] = self.test_user.username
            
            # Create a sample quiz with 3 questions
            questions = [
                {
                    'question': 'Q1?',
                    'options': {'A': 'Opt1', 'B': 'Opt2'},
                    'correct': 'A',
                    'explanation': 'Exp1'
                },
                {
                    'question': 'Q2?',
                    'options': {'A': 'Opt1', 'B': 'Opt2'},
                    'correct': 'B',
                    'explanation': 'Exp2'
                },
                {
                    'question': 'Q3?',
                    'options': {'A': 'Opt1', 'B': 'Opt2'},
                    'correct': 'A',
                    'explanation': 'Exp3'
                }
            ]
            session['questions'] = questions
            session['topic'] = 'Test Topic'
        
        # Submit quiz with only 2 answers (1 correct, 1 incorrect)
        response = self.client.post('/submit', data={
            'topic': 'Test Topic',
            'answer_0': 'A',  # Correct
            'answer_1': 'A'   # Incorrect
            # answer_2 not submitted
        })
        
        self.assertEqual(response.status_code, 200)
        
        # Check that all questions are in results
        self.assertIn(b'Q1?', response.data)
        self.assertIn(b'Q2?', response.data)
        self.assertIn(b'Q3?', response.data)
        
        # Check score calculation (1 correct out of 3 total = 33.3%)
        self.assertIn(b'33.3%', response.data)
        
        # Check that unanswered question is marked
        self.assertIn(b'Not answered', response.data)
        
        # Verify attempt was saved to database
        attempt = QuizAttempt.query.filter_by(user_id=self.test_user.id).first()
        self.assertIsNotNone(attempt)
        self.assertAlmostEqual(attempt.score, 33.3, places=1)

    def test_view_scores_after_five_quizzes(self):
        """Test that scores are visible after taking 5 quizzes"""
        with self.client.session_transaction() as session:
            session['user_id'] = self.test_user.id
            session['username'] = self.test_user.username

        # Add 5 quiz attempts with different scores
        scores = [75, 80, 85, 90, 95]
        for score in scores:
            attempt = QuizAttempt(
                user_id=self.test_user.id,
                score=score,
                topic='Python',
                level='Advanced' if score >= 80 else 'Intermediate'
            )
            db.session.add(attempt)
        db.session.commit()

        # View progress page
        response = self.client.get('/progress')
        self.assertEqual(response.status_code, 200)

        # Check that all scores are shown
        response_text = response.data.decode('utf-8')
        for score in scores:
            self.assertIn(str(score), response_text)

        # Check that average score is shown (85%)
        self.assertIn('85.0', response_text)

if __name__ == "__main__":
    unittest.main()
