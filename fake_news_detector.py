# fake_news_detector.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib

# Download required NLTK data - including punkt_tab for newer NLTK versions
try:
    nltk.download('punkt_tab')  # For newer NLTK versions
except:
    pass
nltk.download('punkt')
nltk.download('stopwords')

class FakeNewsDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Fallback to simple split if NLTK tokenizer fails
            tokens = text.split()
        
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def create_sample_data(self):
        """Create sample dataset for demonstration"""
        # This is sample data - in production, use real datasets like:
        # - Kaggle Fake News Dataset
        # - LIAR dataset
        # - FakeNewsNet dataset
        
        fake_news = [
            "Scientists discover aliens living on Mars, government covering up evidence",
            "Miracle cure found that doctors don't want you to know about",
            "Celebrity dies in car crash, family confirms tragic news",
            "New study shows vaccines cause autism in 90% of children",
            "Local man wins lottery 50 times using this one weird trick",
            "Breaking: World War 3 starts tomorrow, insider sources reveal",
            "Diet pill melts away 50 pounds in one week without exercise",
            "Shocking: Your smartphone is secretly recording everything you say",
            "Incredible footage shows dinosaurs still alive in remote jungle",
            "Government admits to controlling weather with secret technology"
        ]
        
        real_news = [
            "Stock market closes higher amid positive earnings reports",
            "New climate change research published in Nature journal",
            "Local school receives funding for new science laboratory",
            "City council approves budget for infrastructure improvements",
            "University researchers develop new cancer treatment method",
            "Economic indicators show steady growth in manufacturing sector",
            "Environmental protection agency releases new air quality standards",
            "Technology company announces quarterly earnings results",
            "Medical journal publishes study on heart disease prevention",
            "Transportation department plans highway expansion project"
        ]
        
        # Create DataFrame
        data = []
        for text in fake_news:
            data.append({'text': text, 'label': 1})  # 1 = fake
        for text in real_news:
            data.append({'text': text, 'label': 0})  # 0 = real
        
        return pd.DataFrame(data)
    
    def train_model(self, df):
        """Train the fake news detection model"""
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        X = df['processed_text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline with TF-IDF vectorizer and classifier
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        # Train model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        return accuracy
    
    def predict(self, text):
        """Predict if news is fake or real"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet!")
        
        processed_text = self.preprocess_text(text)
        prediction = self.pipeline.predict([processed_text])[0]
        probability = self.pipeline.predict_proba([processed_text])[0]
        
        return {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': float(max(probability)),
            'fake_probability': float(probability[1]),
            'real_probability': float(probability[0])
        }
    
    def save_model(self, filepath='fake_news_model.pkl'):
        """Save trained model to file"""
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='fake_news_model.pkl'):
        """Load trained model from file"""
        self.pipeline = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

# Training script
if __name__ == "__main__":
    # Initialize detector
    detector = FakeNewsDetector()
    
    # Create sample data (replace with real dataset)
    print("Creating sample dataset...")
    df = detector.create_sample_data()
    
    # Train model
    print("Training model...")
    accuracy = detector.train_model(df)
    
    # Save model
    detector.save_model()
    
    # Test predictions
    print("\nTesting predictions:")
    test_articles = [
        "Breaking: Scientists discover cure for all diseases using this simple trick",
        "Local government approves new budget for public transportation improvements"
    ]
    
    for article in test_articles:
        result = detector.predict(article)
        print(f"\nArticle: {article[:50]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}")