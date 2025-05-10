# SmartLearn - AI-Powered Quiz Generator

An interactive web application that generates quizzes on any topic using OpenAI's GPT-3.5 Turbo, provides instant feedback, and recommends learning resources based on performance.

## System Architecture

```mermaid
graph TD
    %% Frontend Components
    subgraph Frontend[User Interface]
        UI[Web Browser] --> |1. Enter Topic| Form[Quiz Form]
        Form --> |2. Display Quiz| Questions[Quiz Questions]
        Questions --> |3. Submit Answers| Results[Results Page]
    end

    %% Backend Components
    subgraph Backend[Flask Application]
        Server[Flask Server] --> |4. Process Request| Generator[Quiz Generator]
        Generator --> |5. API Call| OpenAI[OpenAI GPT-3.5]
        OpenAI --> |6. Quiz Content| Generator
        Server --> |7. Process Answers| Calculator[Score Calculator]
        Calculator --> |8. Calculate Level| Analyzer[Level Analyzer]
    end

    %% External Services
    subgraph External[External Services]
        YouTube[YouTube API]
    end

    %% Data Flow
    Form --> |Request| Server
    Generator --> |Quiz Data| Questions
    Results --> |Score| Calculator
    Analyzer --> |Learning Level| Results
    Results --> |9. Generate Links| YouTube

    %% Styling
    classDef frontend fill:#d4f1f9,stroke:#0077b6
    classDef backend fill:#e9ecef,stroke:#495057
    classDef external fill:#ffe5d9,stroke:#bc4749

    class UI,Form,Questions,Results frontend
    class Server,Generator,Calculator,Analyzer backend
    class OpenAI,YouTube external
```

## Application Components

### Frontend
- **Quiz Form**: Accepts topic input from users
- **Quiz Questions**: Displays multiple-choice questions
- **Results Page**: Shows score, proficiency level, and learning resources

### Backend
- **Flask Server**: Handles HTTP requests and session management
- **Quiz Generator**: Interfaces with OpenAI to create quiz content
- **Score Calculator**: Processes user answers and calculates scores
- **ML Classifier**: Gradient Boosting model that predicts user proficiency level based on:
  - Average quiz score
  - Number of quizzes taken
  - Hint usage patterns
  - Retry attempts

### External Services
- **OpenAI GPT-3.5**: Generates quiz questions and answers
- **YouTube**: Provides relevant tutorial recommendations
        L[OpenAI GPT-3.5] --> I
        M[YouTube] --> F
    end

    style Frontend fill:#f9f,stroke:#333,stroke-width:2px
    style Backend fill:#bbf,stroke:#333,stroke-width:2px
    style External fill:#bfb,stroke:#333,stroke-width:2px
```

## Features

- **Dynamic Quiz Generation**: Create quizzes on any topic using AI
- **Multiple Choice Format**: Each quiz includes multiple-choice questions with four options
- **Instant Feedback**: Get immediate results after quiz submission
- **Performance Analysis**: 
  - Score calculation
  - Proficiency level assessment (Beginner/Intermediate/Expert)
  - Detailed feedback for each question
- **Learning Resources**: Personalized YouTube tutorial recommendations based on proficiency level
- **User-Friendly Interface**: Clean, responsive design using Bootstrap

  ## Prerequisites
  # - Python 3.x: This project was developed and tested using Python 3.x.
  # - OpenAI API key: This project uses OpenAI's GPT-3.5 model to generate quizzes.

## Prerequisites

- Python 3.x
- OpenAI API key

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd PythonProject4
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5006
   ```

3. Enter any topic you'd like to be quizzed on and click "Generate Quiz"

## User Classification

The application uses a Gradient Boosting Classifier to determine user proficiency levels based on multiple performance metrics:

### Features Used
- Average quiz score
- Number of quizzes taken
- Hint usage patterns
- Retry attempts

### Proficiency Levels
- **Advanced**: Demonstrates mastery of topics
- **Intermediate**: Shows good understanding with room for improvement
- **Beginner**: Recommended to start with basic tutorials

### Model Performance
- **Accuracy**: 100% on test set
- **Cross-validation Score**: 97.9% (±2.6%)
- **Feature Scaling**: StandardScaler for consistent predictions

### Fallback Classification
If the ML model is unavailable, the system falls back to a simple score-based classification:
- **Advanced** (≥80%): Advanced understanding
- **Intermediate** (50-79%): Good foundation
- **Beginner** (<50%): Needs fundamental review

## Technical Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, Bootstrap 5
- **AI Integration**: OpenAI GPT-3.5 Turbo
- **Machine Learning**: scikit-learn (Gradient Boosting Classifier)
- **Data Processing**: pandas, numpy
- **Session Management**: Flask-Session
- **Environment Management**: python-dotenv

## File Structure

- `app.py`: Main Flask application and route handlers
- `quiz_generator.py`: OpenAI integration and quiz generation logic
- `user_performance_classifier.py`: ML model training and prediction
- `templates/`: HTML templates
  - `quiz.html`: Quiz interface
  - `results.html`: Results and recommendations page
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (API keys)
- `user_level_classifier.joblib`: Trained ML model
- `user_level_scaler.joblib`: Feature scaler for ML input



