📧 Spam Mail Recognition using Machine Learning
🔍 Project Overview

This project focuses on detecting spam emails using Machine Learning and Natural Language Processing (NLP) techniques.
The model analyzes the content of an email and predicts whether it is Spam or Not Spam, achieving high accuracy and efficiency.

🧠 Key Features

✅ Text preprocessing using NLTK (tokenization, stopword removal, stemming)
✅ Feature extraction using TF-IDF Vectorization
✅ Model training with Logistic Regression, Naive Bayes, and Random Forest
✅ Evaluation and accuracy comparison between models
✅ Real-time user input prediction (enter any message to test spam detection)

⚙️ Tech Stack

Programming Language: Python

Libraries Used:

pandas, numpy – Data handling and preprocessing

nltk – Text preprocessing (stopwords, stemming)

scikit-learn – Model training, evaluation, and TF-IDF Vectorizer

matplotlib, seaborn – Data visualization

joblib – Model saving and loading

🧩 Workflow
1️⃣ Data Loading

Dataset: SMS Spam Collection Dataset (Kaggle)

2️⃣ Data Cleaning

Remove null values

Normalize text (lowercase, remove punctuation)

3️⃣ NLP Preprocessing

Tokenization

Stopword removal

Word stemming using PorterStemmer

4️⃣ Feature Extraction

Convert text to numerical format using TF-IDF Vectorization

5️⃣ Model Training & Evaluation

Trained multiple ML models and compared accuracies:

Logistic Regression → 98.9% Accuracy 🏆

Naive Bayes → 97.6%

Random Forest → 96.8%

6️⃣ Real-Time Testing

Allows user to input a custom message and see live prediction output:

Input: "Congratulations! You have won a free iPhone!"
Prediction: 🚫 SPAM

📊 Results
Model	Accuracy
Logistic Regression	98.9%
Naive Bayes	97.6%
Random Forest	96.8%
📁 Folder Structure
SpamMailRecognition/
│
├── neee.py                 # Main Python script
├── spam.csv                # Dataset file
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── venv/                   # Virtual environment (optional)

🧾 Sample Output
STEP 1: LOADING DATA
✓ Dataset loaded successfully!

STEP 2: PREPROCESSING
✓ Text cleaned and vectorized

STEP 3: MODEL TRAINING
🏆 Best Model: Logistic Regression
Accuracy: 98.9%

🧠 Learnings

Gained practical knowledge of text preprocessing and NLP

Improved understanding of classification algorithms

Learned model evaluation and performance tuning

Implemented real-time prediction feature

🚀 Future Enhancements

🔹 Deploy the model using Streamlit / Flask Web App
🔹 Add deep learning models like LSTM or BERT
🔹 Create a REST API for integration with email platforms

👨‍💻 Author

Aryan Bajpai
📧 aryanbajpai531@gmail.com
