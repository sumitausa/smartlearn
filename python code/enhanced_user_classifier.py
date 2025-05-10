import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
from datetime import datetime, timedelta

def generate_synthetic_user_data(num_users=300, random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)
    data = []

    # Define user archetypes with expanded characteristics
    archetypes = {
        'Advanced': {
            'score_mean': 88, 'score_std': 8,
            'quiz_mean': 12, 'quiz_std': 3,
            'hint_mean': 0.15, 'hint_std': 0.1,
            'retry_mean': 0.8, 'retry_std': 0.5,
            'session_duration_mean': 45,  # minutes
            'topics_explored_mean': 8,
            'completion_rate_mean': 0.95
        },
        'Intermediate': {
            'score_mean': 75, 'score_std': 10,
            'quiz_mean': 8, 'quiz_std': 3,
            'hint_mean': 0.4, 'hint_std': 0.15,
            'retry_mean': 1.5, 'retry_std': 0.8,
            'session_duration_mean': 30,
            'topics_explored_mean': 5,
            'completion_rate_mean': 0.85
        },
        'Beginner': {
            'score_mean': 55, 'score_std': 12,
            'quiz_mean': 5, 'quiz_std': 2,
            'hint_mean': 0.7, 'hint_std': 0.2,
            'retry_mean': 2.5, 'retry_std': 1.0,
            'session_duration_mean': 20,
            'topics_explored_mean': 3,
            'completion_rate_mean': 0.70
        }
    }

    # Topics for simulation
    topics = ['Python Basics', 'Data Types', 'Control Flow', 'Functions', 
              'OOP', 'File Handling', 'Error Handling', 'Modules', 
              'Data Structures', 'Algorithms']

    start_date = datetime.now() - timedelta(days=90)

    for user_id in range(num_users):
        # Generate unique user identifier
        username = f"user_{user_id:04d}"
        
        # Randomly select initial archetype
        archetype = np.random.choice(
            ['Beginner', 'Intermediate', 'Advanced'],
            p=[0.5, 0.35, 0.15]
        )
        arch = archetypes[archetype]

        # Generate base metrics
        avg_score = np.clip(np.random.normal(
            arch['score_mean'], arch['score_std']
        ), 0, 100)
        
        hint_usage = np.clip(np.random.normal(
            arch['hint_mean'], arch['hint_std']
        ), 0, 1)

        num_quizzes = max(1, int(np.random.normal(
            arch['quiz_mean'], arch['quiz_std']
        )))

        # Generate additional user-specific features
        topics_explored = random.sample(topics, 
            min(len(topics), int(arch['topics_explored_mean'] * (1 + random.uniform(-0.2, 0.2))))
        )
        
        completion_rate = min(1.0, arch['completion_rate_mean'] * (1 + random.uniform(-0.1, 0.1)))
        
        avg_session_duration = max(5, int(arch['session_duration_mean'] * 
            (1 + random.uniform(-0.2, 0.2))))

        # Generate activity dates
        last_active = start_date + timedelta(
            days=random.randint(0, 90),
            hours=random.randint(0, 23)
        )

        # Calculate user engagement score
        engagement_score = (
            (completion_rate * 0.4) +
            (min(1.0, num_quizzes/10) * 0.3) +
            (min(1.0, len(topics_explored)/len(topics)) * 0.3)
        ) * 100

        # Determine final level based on comprehensive metrics
        if avg_score > 82 and hint_usage < 0.25 and engagement_score > 80:
            level = 'Advanced'
        elif avg_score > 60 and hint_usage < 0.6 and engagement_score > 50:
            level = 'Intermediate'
        else:
            level = 'Beginner'

        data.append({
            'user_id': username,
            'avg_score': avg_score,
            'num_quizzes': num_quizzes,
            'hint_usage': hint_usage,
            'topics_explored': len(topics_explored),
            'completion_rate': completion_rate,
            'avg_session_duration': avg_session_duration,
            'engagement_score': engagement_score,
            'days_since_last_active': (datetime.now() - last_active).days,
            'level': level
        })

    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate enhanced user data
    df = generate_synthetic_user_data()
    
    # Display sample users with their characteristics
    print("\nSample User Profiles:")
    print("=" * 80)
    sample_users = df.sample(5)
    for _, user in sample_users.iterrows():
        print(f"\nUser ID: {user['user_id']}")
        print(f"Level: {user['level']}")
        print(f"Average Score: {user['avg_score']:.1f}%")
        print(f"Quizzes Completed: {user['num_quizzes']}")
        print(f"Topics Explored: {user['topics_explored']}")
        print(f"Engagement Score: {user['engagement_score']:.1f}/100")
        print(f"Last Active: {user['days_since_last_active']} days ago")
        print("-" * 40)
