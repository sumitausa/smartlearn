# quiz_generator.py
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def generate_quiz(topic, num_questions=3, difficulty="Intermediate"):
    # Create a prompt for GPT to generate quiz questions
    prompt = f"""Generate a quiz about {topic} with EXACTLY {num_questions} multiple-choice questions at {difficulty} level.
    For each question, provide:
    1. A clear, specific question
    2. Four options labeled A, B, C, D
    3. The correct answer letter
    4. A brief explanation of why that answer is correct
    
    Format the response as a JSON object with this structure:
    {{
        "questions": [
            {{
                "question": "Question text here",
                "options": {{
                    "A": "First option",
                    "B": "Second option",
                    "C": "Third option",
                    "D": "Fourth option"
                }},
                "correct": "A",
                "explanation": "Explanation here"
            }}
        ]
    }}
    
    IMPORTANT: Return EXACTLY {num_questions} questions, no more and no less.
    """

    try:
        # Call OpenAI API to generate questions
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[{
                "role": "system",
                "content": "You are a knowledgeable teacher creating quiz questions. Generate clear, accurate, and engaging questions suitable for the specified difficulty level. Always generate exactly the number of questions requested."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=4000
        )

        # Extract the quiz content from the response
        quiz_content = response.choices[0].message.content
        
        # Convert the response to a Python dictionary
        import json
        try:
            quiz_data = json.loads(quiz_content)
            # Ensure exactly num_questions are returned
            if 'questions' in quiz_data and len(quiz_data['questions']) != num_questions:
                print(f"Warning: Got {len(quiz_data['questions'])} questions instead of {num_questions}")
                quiz_data['questions'] = quiz_data['questions'][:num_questions]
            print("Generated quiz data:", quiz_data)
            return quiz_data
        except json.JSONDecodeError:
            print("Error: Could not parse quiz data as JSON")
            return generate_fallback_quiz(topic, num_questions)
            
    except Exception as e:
        print(f"Error generating quiz: {str(e)}")
        return generate_fallback_quiz(topic)

def generate_fallback_quiz(topic):
    """Generate a fallback quiz if the API call fails"""
    return {
        'questions': [
            {
                'index': 1,
                'question': f'What is {topic}?',
                'options': [
                    {'label': 'A', 'text': 'A programming language'},
                    {'label': 'B', 'text': 'A development tool'},
                    {'label': 'C', 'text': 'A technology concept'},
                    {'label': 'D', 'text': 'All of the above'}
                ],
                'correct_answer': 'D',
                'explanation': 'This is a general fallback question.'
            }
        ]
    }
