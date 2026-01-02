🚍 BusBuddy – Smart Bus Assistant Chatbot

BusBuddy is an intelligent bus assistant chatbot that helps users with bus fare queries, route availability, and general transport-related questions.
It uses Natural Language Processing (NLP), Machine Learning, and Flask to provide accurate and user-friendly responses through a web interface.

✨ Features
🤖 Chatbot Interface for natural language queries

💰 Bus Fare Prediction (Adult & Child fares)

🚌 Route Availability & Direct Bus Detection

🔁 Transfer Route Suggestions

🔍 Fuzzy Matching for stop name errors

🌐 Web-based UI using Flask & HTML

⚡ Upgraded to Python 3.11 & TensorFlow 2.15

🛠️ Tech Stack
Backend: Python, Flask

Machine Learning: TensorFlow (Keras), Scikit-learn

NLP: NLTK

Data Handling: Pandas, NumPy

Matching: FuzzyWuzzy

Frontend: HTML, CSS, JavaScript

📂 Project Structure
bus-buddy-main/

│
├── chatbot.py  # Main Flask application

├── train_chatbot.py             # Chatbot training script

├── intents.json                 # Chatbot intents & responses

├── chatbot_model.keras          # Trained chatbot model

├── words.pkl                    # NLP vocabulary

├── classes.pkl                  # Intent classes

├── fare_prediction_model.pkl    # Fare prediction ML model

├── surat_bus.csv                # Bus route data

├── SURAT5.csv                   # Fare dataset

├── templates/
│   └── index.html               # Chat UI

├── requirements.txt             # Project dependencies

└── README.md                    # Project documentation

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/owais1724/bus-buddy.git
cd bus-buddy
2️⃣ Create Virtual Environment (Python 3.11 recommended)
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install tensorflow==2.15.0 numpy==1.26.4 scikit-learn==1.5.2 flask pandas nltk fuzzywuzzy python-Levenshtein
4️⃣ Download NLTK Data (first time only)
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
5️⃣ Train Chatbot Model (Optional – if model not present)
python train_chatbot.py
6️⃣ Run the Application
python chatbot.py
Open your browser and visit:

http://127.0.0.1:5000/
💬 Example Queries
Hi

Fare from Adajan to Vesu

Which bus goes from Citylight to Udhna

Bus route from Katargam to Varachha

Thanks

🧠 How It Works
User enters a message through the web UI

NLP preprocessing converts text into numerical features

Trained ML model predicts the intent

Fare & route logic is applied if required

Bot responds with accurate information

🎓 Academic Relevance
This project demonstrates:

NLP-based chatbot design

ML model deployment

Real-world data handling

Flask-based web integration

✅ Suitable for Final Year Project / Mini Project

🔮 Future Enhancements
Real-time bus tracking

Database integration

Voice-based interaction

Mobile application version

Multi-city support

👨‍💻 Authors
Syed Owais


📜 License
This project is for educational purposes only.

