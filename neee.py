
import pandas as pd
import numpy as np
import re
import string
import warnings
warnings.filterwarnings('ignore')


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)


import joblib
import pickle


print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


DATASET_PATH = 'spam.csv'
MODEL_SAVE_PATH = 'best_spam_model.pkl'
VECTORIZER_SAVE_PATH = 'tfidf_vectorizer.pkl'
RANDOM_STATE = 42
TEST_SIZE = 0.2



def load_data(filepath):
    """
    Load the spam dataset from CSV file.
    
    Args:
        filepath (str): Path to the spam.csv file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    try:
        
        df = pd.read_csv(filepath, encoding='latin-1')
        
        
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
        
        print(f"✓ Dataset loaded successfully!")
        print(f"  Total samples: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        return df
    
    except FileNotFoundError:
        print(f"✗ Error: File '{filepath}' not found!")
        print("  Please ensure spam.csv is in the same directory.")
        return None



def handle_missing_values(df):
    """
    Check and handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("\n" + "="*70)
    print("STEP 2: HANDLING MISSING VALUES")
    print("="*70)
    
    print(f"Missing values before cleaning:")
    print(df.isnull().sum())
    
    
    df = df.dropna()
    
    
    initial_size = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_size - len(df)
    
    print(f"\n✓ Missing values handled!")
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Final dataset size: {len(df)}")
    
    return df


def encode_labels(df):
    """
    Encode labels: spam = 1, ham = 0.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with encoded labels
    """
    print("\n" + "="*70)
    print("STEP 3: ENCODING LABELS")
    print("="*70)
    
    
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    
    print(f"✓ Labels encoded successfully!")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nPercentage:")
    print(df['label'].value_counts(normalize=True) * 100)
    
    return df


def clean_text(text):
    """
    Clean and preprocess text data.
    
    Steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Remove numbers
    4. Remove extra whitespaces
    5. Remove stopwords
    6. Apply stemming
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    
    text = text.lower()
    
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    
    text = re.sub(r'\d+', '', text)
    
    
    text = ' '.join(text.split())
    
   
    words = text.split()
    
    
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
   
    return ' '.join(words)


def preprocess_text_data(df):
    """
    Apply text cleaning to all messages in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with cleaned messages
    """
    print("\n" + "="*70)
    print("STEP 4: TEXT PREPROCESSING")
    print("="*70)
    print("Cleaning text data (this may take a moment)...")
    
    
    df['cleaned_message'] = df['message'].apply(clean_text)
    
    print(f"✓ Text preprocessing completed!")
    print(f"\nExample transformations:")
    for i in range(min(3, len(df))):
        print(f"\nOriginal: {df['message'].iloc[i][:80]}...")
        print(f"Cleaned:  {df['cleaned_message'].iloc[i][:80]}...")
    
    return df



def extract_features(X_train, X_test):
    """
    Extract features using TF-IDF Vectorizer.
    
    Args:
        X_train (pd.Series): Training messages
        X_test (pd.Series): Testing messages
        
    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, vectorizer)
    """
    print("\n" + "="*70)
    print("STEP 5: FEATURE EXTRACTION (TF-IDF)")
    print("="*70)
    
    
    vectorizer = TfidfVectorizer(max_features=3000, min_df=2, max_df=0.8)
    
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"✓ Feature extraction completed!")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Training features shape: {X_train_tfidf.shape}")
    print(f"  Testing features shape: {X_test_tfidf.shape}")
    
    return X_train_tfidf, X_test_tfidf, vectorizer



def train_models(X_train, y_train):
    """
    Train multiple classification models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        dict: Dictionary of trained models
    """
    print("\n" + "="*70)
    print("STEP 6: MODEL TRAINING")
    print("="*70)
    
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'SVM': SVC(kernel='linear', random_state=RANDOM_STATE)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✓ {name} trained successfully!")
    
    return trained_models



def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models and print metrics.
    
    Args:
        models (dict): Dictionary of trained models
        X_test: Testing features
        y_test: Testing labels
        
    Returns:
        tuple: (best_model_name, best_model, best_accuracy)
    """
    print("\n" + "="*70)
    print("STEP 7: MODEL EVALUATION")
    print("="*70)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"{name}")
        print('='*70)
        
        
        y_pred = model.predict(X_test)
        
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
        
        print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Ham    Spam")
        print(f"Actual Ham    {cm[0][0]:<6} {cm[0][1]:<6}")
        print(f"       Spam   {cm[1][0]:<6} {cm[1][1]:<6}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']
    best_model = models[best_model_name]
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"   Precision: {results[best_model_name]['precision']:.4f}")
    print(f"   Recall: {results[best_model_name]['recall']:.4f}")
    print(f"   F1-Score: {results[best_model_name]['f1_score']:.4f}")
    
    return best_model_name, best_model, best_accuracy



def save_model(model, vectorizer, model_path, vectorizer_path):
    """
    Save the trained model and vectorizer to disk.
    
    Args:
        model: Trained model
        vectorizer: Fitted TF-IDF vectorizer
        model_path (str): Path to save the model
        vectorizer_path (str): Path to save the vectorizer
    """
    print("\n" + "="*70)
    print("STEP 8: SAVING MODEL")
    print("="*70)
    
    
    joblib.dump(model, model_path)
    print(f"✓ Model saved to: {model_path}")
    

    joblib.dump(vectorizer, vectorizer_path)
    print(f"✓ Vectorizer saved to: {vectorizer_path}")


def load_saved_model(model_path, vectorizer_path):
    """
    Load a saved model and vectorizer from disk.
    
    Args:
        model_path (str): Path to the saved model
        vectorizer_path (str): Path to the saved vectorizer
        
    Returns:
        tuple: (model, vectorizer)
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print(f"✓ Model and vectorizer loaded successfully!")
    return model, vectorizer



def predict_spam(message, model, vectorizer):
    """
    Predict whether a message is spam or ham.
    
    Args:
        message (str): Input message
        model: Trained model
        vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        tuple: (prediction, probability)
    """
    
    cleaned_message = clean_text(message)
    
    
    message_tfidf = vectorizer.transform([cleaned_message])
    
    
    prediction = model.predict(message_tfidf)[0]
    
    
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(message_tfidf)[0]
        spam_probability = probability[1]
    else:
        
        spam_probability = None
    
    return prediction, spam_probability


def interactive_prediction(model, vectorizer):
    """
    Interactive function for users to test custom messages.
    
    Args:
        model: Trained model
        vectorizer: Fitted TF-IDF vectorizer
    """
    print("\n" + "="*70)
    print("INTERACTIVE SPAM DETECTION")
    print("="*70)
    print("\nEnter messages to check if they're spam or ham.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        message = input("\nEnter a message: ").strip()
        
        if message.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using Spam Detection!")
            break
        
        if not message:
            print("Please enter a valid message.")
            continue
        
        prediction, probability = predict_spam(message, model, vectorizer)
        
        label = "🚫 SPAM" if prediction == 1 else "✓ HAM (Not Spam)"
        
        print(f"\n{'='*70}")
        print(f"Result: {label}")
        
        if probability is not None:
            print(f"Confidence: {probability*100:.2f}% spam")
        
        print('='*70)



def main():
    """
    Main function to execute the complete ML pipeline.
    """
    print("\n" + "="*70)
    print("  SPAM MAIL RECOGNITION USING MACHINE LEARNING")
    print("="*70)
    
    
    df = load_data(DATASET_PATH)
    if df is None:
        return
    
    
    df = handle_missing_values(df)
    
    
    df = encode_labels(df)
    
    
    df = preprocess_text_data(df)
    
    
    X = df['cleaned_message']
    y = df['label']
    
    
    print("\n" + "="*70)
    print("SPLITTING DATA (80% Train, 20% Test)")
    print("="*70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Testing samples: {len(X_test)}")
    
    
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)
    
    
    trained_models = train_models(X_train_tfidf, y_train)
    
    
    best_model_name, best_model, best_accuracy = evaluate_models(
        trained_models, X_test_tfidf, y_test
    )
    
    
    save_model(best_model, vectorizer, MODEL_SAVE_PATH, VECTORIZER_SAVE_PATH)
    
    
    print("\n" + "="*70)
    print("TEST YOUR OWN MESSAGES")
    print("="*70)
    
    
    test_messages = [
        "Congratulations! You've won a free iPhone. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT! Your account will be suspended. Verify now at http://fake-link.com",
        "Can you pick up some milk on your way home?",
    ]
    
    print("\nTesting with example messages:\n")
    for msg in test_messages:
        prediction, probability = predict_spam(msg, best_model, vectorizer)
        label = "SPAM" if prediction == 1 else "HAM"
        prob_str = f" ({probability*100:.1f}%)" if probability is not None else ""
        print(f"Message: {msg[:60]}...")
        print(f"Prediction: {label}{prob_str}\n")
    
    
    user_input = input("\nWould you like to test your own messages? (yes/no): ").strip().lower()
    if user_input in ['yes', 'y']:
        interactive_prediction(best_model, vectorizer)
    
    print("\n" + "="*70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nModel saved as: {MODEL_SAVE_PATH}")
    print(f"Vectorizer saved as: {VECTORIZER_SAVE_PATH}")
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {best_accuracy*100:.2f}%")
    print("\nYou can load and use the model later with:")
    print("  model, vectorizer = load_saved_model('best_spam_model.pkl', 'tfidf_vectorizer.pkl')")
    print("  predict_spam('your message', model, vectorizer)")



if __name__ == "__main__":
    main()