import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class FakeNewsDetector:
    def __init__(self):
        self.pipeline = None
        self.stop_words = set(stopwords.words('english'))
        self._compile_regex()

    def _compile_regex(self):
        """Compile regex patterns for efficiency"""
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.html_pattern = re.compile(r'<.*?>')
        self.special_char_pattern = re.compile(r'[^a-zA-Z\s]')
        self.whitespace_pattern = re.compile(r'\s+')

    def preprocess_text(self, text):
        """Clean and prepare text for analysis"""
        if not isinstance(text, str):
            return ""
            
        # Clean text
        text = text.lower()
        text = self.url_pattern.sub('', text)
        text = self.html_pattern.sub('', text)
        text = self.special_char_pattern.sub('', text)
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
            
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        return ' '.join(tokens)

    def create_sample_data(self, size=500):
        """Generate training data"""
        fake_samples = [
            "Scientists discover miracle cure doctors don't want you to know",
            "Celebrity dies in shocking accident caught on camera",
            "Government hiding alien technology in secret facility"
        ]
        
        real_samples = [
            "New study shows benefits of regular exercise",
            "City council approves new park construction",
            "Economy shows steady growth this quarter"
        ]
        
        # Augment data
        data = []
        for i in range(size//2):
            for sample in fake_samples:
                data.append({'text': f"{sample} {i}", 'label': 1})
            for sample in real_samples:
                data.append({'text': f"{sample} {i}", 'label': 0})
                
        return pd.DataFrame(data)

    def train_model(self, df=None):
        """Train the detection model"""
        if df is None:
            df = self.create_sample_data()
            
        df['processed'] = df['text'].apply(self.preprocess_text)
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed'], df['label'], test_size=0.2, random_state=42
        )
        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(classification_report(y_test, y_pred))
        
        return accuracy_score(y_test, y_pred)

    def predict(self, text):
        """Make a prediction on new text"""
        if self.pipeline is None:
            raise ValueError("Model not trained")
            
        processed = self.preprocess_text(text)
        pred = self.pipeline.predict([processed])[0]
        proba = self.pipeline.predict_proba([processed])[0]
        
        return {
            'prediction': 'Fake' if pred == 1 else 'Real',
            'confidence': float(max(proba)),
            'fake_probability': float(proba[1]),
            'real_probability': float(proba[0]),
            'text_processed': processed
        }

    def save_model(self, path):
        """Save model with version info"""
        save_data = {
            'pipeline': self.pipeline,
            'versions': {
                'numpy': np.__version__,
                'sklearn': sklearn.__version__
            }
        }
        joblib.dump(save_data, path)

def load_model(self, path):
    """Load model with version checking"""
    loaded = joblib.load(path)
    self.pipeline = loaded['pipeline']
    
    # Log version info
    print(f"Model was saved with:")
    print(f"- NumPy: {loaded['versions'].get('numpy', 'unknown')}")
    print(f"- scikit-learn: {loaded['versions'].get('sklearn', 'unknown')}")
    print(f"Current versions:")
    print(f"- NumPy: {np.__version__}")
    print(f"- scikit-learn: {sklearn.__version__}")

def load_model(self, path):
    """Load trained model"""
    self.pipeline = joblib.load(path)