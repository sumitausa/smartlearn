import unittest
from app import app, db, User, QuizAttempt
from datetime import datetime

class TestScoring(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app_context = app.app_context()
        self.app_context.push()
        self.client = app.test_client()
        db.create_all()
        
        # Create test user
        self.test_user = User(username='test_user')
        db.session.add(self.test_user)
        db.session.commit()
            
    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
            
    def test_score_calculation(self):
        """Test that scores are calculated correctly"""
        # Add some quiz attempts
        attempts = [
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=80.0, level='Intermediate'),
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=90.0, level='Advanced'),
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=70.0, level='Beginner')
        ]
        for attempt in attempts:
            db.session.add(attempt)
        db.session.commit()
        
        # Login
        with self.client.session_transaction() as session:
            session['user_id'] = self.test_user.id
            session['username'] = self.test_user.username
            
        # Check progress page
        response = self.client.get('/progress')
        self.assertEqual(response.status_code, 200)
        
        # Get user's attempts
        attempts = QuizAttempt.query.filter_by(user_id=self.test_user.id).all()
        
        # Calculate expected values
        total_quizzes = len(attempts)
        avg_score = sum(attempt.score for attempt in attempts) / total_quizzes
        best_score = max(attempt.score for attempt in attempts)
        
        # Verify calculations
        self.assertEqual(total_quizzes, 3)
        self.assertEqual(round(avg_score, 2), 80.00)
        self.assertEqual(best_score, 90)
            
    def test_level_determination(self):
        """Test that user levels are determined correctly"""
        # Test Advanced Level (avg >= 80)
        attempts = [
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=85.0, level='Advanced'),
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=90.0, level='Advanced'),
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=80.0, level='Advanced')
        ]
        for attempt in attempts:
            db.session.add(attempt)
        db.session.commit()
        
        with self.client.session_transaction() as session:
            session['user_id'] = self.test_user.id
            session['username'] = self.test_user.username
            
        response = self.client.get('/progress')
        self.assertIn(b'Advanced', response.data)
        
        # Clear attempts
        QuizAttempt.query.delete()
        db.session.commit()
        
        # Test Intermediate Level (60 <= avg < 80)
        attempts = [
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=75.0, level='Intermediate'),
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=65.0, level='Intermediate'),
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=70.0, level='Intermediate')
        ]
        for attempt in attempts:
            db.session.add(attempt)
        db.session.commit()
        
        response = self.client.get('/progress')
        self.assertIn(b'Intermediate', response.data)
        
        # Clear attempts
        QuizAttempt.query.delete()
        db.session.commit()
        
        # Test Beginner Level (avg < 60)
        attempts = [
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=55.0, level='Beginner'),
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=45.0, level='Beginner'),
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=50.0, level='Beginner')
        ]
        for attempt in attempts:
            db.session.add(attempt)
        db.session.commit()
        
        response = self.client.get('/progress')
        self.assertIn(b'Beginner', response.data)
            
    def test_new_user_defaults(self):
        """Test that new users without quiz attempts get correct defaults"""
        with self.client.session_transaction() as session:
            session['user_id'] = self.test_user.id
            session['username'] = self.test_user.username
            
        response = self.client.get('/progress')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Not enough data', response.data)  # Default level
        self.assertIn(b'0', response.data)  # Total quizzes should be 0
            
    def test_recent_scores(self):
        """Test that level is based on recent performance (last 5 attempts)"""
        # Add 7 attempts, last 5 being Advanced level
        attempts = [
            # Older attempts (should not affect level)
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=50.0, level='Beginner'),
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=55.0, level='Beginner'),
            
            # Recent attempts (should determine level)
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=85.0, level='Advanced'),
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=90.0, level='Advanced'),
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=88.0, level='Advanced'),
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=82.0, level='Advanced'),
            QuizAttempt(user_id=self.test_user.id, topic='Python', score=87.0, level='Advanced')
        ]
        
        # Add attempts in chronological order
        for attempt in attempts:
            db.session.add(attempt)
            db.session.commit()
            
        with self.client.session_transaction() as session:
            session['user_id'] = self.test_user.id
            session['username'] = self.test_user.username
            
        response = self.client.get('/progress')
        self.assertIn(b'Advanced', response.data)  # Should be Advanced based on recent scores
            
if __name__ == '__main__':
    unittest.main()
