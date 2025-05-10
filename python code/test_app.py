import requests
from bs4 import BeautifulSoup
import json
import time

def test_all_routes():
    print("\n=== Starting Full App Test ===\n")
    base_url = "http://127.0.0.1:5001"
    s = requests.Session()
    
    def print_section(name):
        print(f"\n{'='*20} {name} {'='*20}\n")
    
    def check_response(response, route):
        print(f"Status code for {route}: {response.status_code}")
        if response.status_code != 200:
            print(f"Error content: {response.text}")
            return False
        return True
    
    # Test 1: Login
    print_section("Testing Login")
    login_data = {"username": "test_user_" + str(int(time.time()))}
    response = s.post(f"{base_url}/login", data=login_data)
    if not check_response(response, "/login"):
        return
    print(f"Logged in as: {login_data['username']}")
    
    # Test 2: Generate Quiz
    print_section("Testing Quiz Generation")
    quiz_data = {"topic": "Python basics"}
    response = s.post(f"{base_url}/generate", data=quiz_data)
    if not check_response(response, "/generate"):
        return
        
    # Parse quiz form
    soup = BeautifulSoup(response.text, 'html.parser')
    form = soup.find('form', {'action': '/submit'})
    if not form:
        print("Error: Could not find quiz form!")
        return
    
    # Collect all questions and prepare submission data
    submit_data = {'topic': quiz_data['topic']}
    radio_inputs = form.find_all('input', {'type': 'radio'})
    question_counts = {}
    
    for radio in radio_inputs:
        name = radio.get('name')
        if name:
            if name not in question_counts:
                question_counts[name] = []
            question_counts[name].append(radio.get('value'))
    
    # Select first option (A) for each question
    for name in question_counts:
        submit_data[name] = question_counts[name][0]
    
    print(f"Found {len(question_counts)} questions")
    print("Submit data:", json.dumps(submit_data, indent=2))
    
    # Test 3: Submit Quiz
    print_section("Testing Quiz Submission")
    response = s.post(f"{base_url}/submit", data=submit_data)
    if not check_response(response, "/submit"):
        return
    print("Quiz submitted successfully")
    
    # Test 4: Check Progress Page
    print_section("Testing Progress Page")
    response = s.get(f"{base_url}/progress")
    if not check_response(response, "/progress"):
        return
    
    # Parse progress page to verify content
    soup = BeautifulSoup(response.text, 'html.parser')
    progress_data = {}
    
    # Look for specific elements
    for div in soup.find_all('div', class_='card-body'):
        text = div.get_text()
        if 'Total Quizzes:' in text:
            progress_data['total_quizzes'] = text
        elif 'Average Score:' in text:
            progress_data['avg_score'] = text
        elif 'Best Score:' in text:
            progress_data['best_score'] = text
        elif 'Current Level:' in text:
            progress_data['level'] = text
    
    print("\nProgress Page Data:")
    for key, value in progress_data.items():
        print(f"{key}: {value.strip()}")
    
    # Test 5: Logout
    print_section("Testing Logout")
    response = s.get(f"{base_url}/logout")
    if not check_response(response, "/logout"):
        return
    print("Logged out successfully")
    
    print("\n=== Test Complete ===\n")
    
if __name__ == "__main__":
    test_all_routes()
