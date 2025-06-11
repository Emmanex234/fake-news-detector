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
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Download required NLTK data
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.download(resource)
        except:
            print(f"Warning: Failed to download NLTK resource {resource}")

download_nltk_resources()

class FakeNewsDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.stop_words = set(stopwords.words('english'))
        self._initialize_text_cleaning_patterns()
    
    def _initialize_text_cleaning_patterns(self):
        """Compile regex patterns for better performance"""
        self.special_char_pattern = re.compile(r'[^a-zA-Z\s]')
        self.whitespace_pattern = re.compile(r'\s+')
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.html_pattern = re.compile(r'<.*?>')
    
    def preprocess_text(self, text):
        """Clean and preprocess text data with enhanced cleaning"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove HTML tags
        text = self.html_pattern.sub('', text)
        
        # Remove special characters and digits
        text = self.special_char_pattern.sub('', text)
        
        # Remove extra whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        # Tokenize and remove stopwords
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Lemmatization could be added here for better results
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def create_sample_data(self, sample_size=1000):
        """Create enhanced sample dataset with more realistic examples"""
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
        
        # Augment dataset by repeating and slightly modifying examples
        augmented_fake = []
        augmented_real = []
        
        for i in range(sample_size // 20):
            for example in fake_news:
                augmented_fake.append(example + f" (variation {i})")
            for example in real_news:
                augmented_real.append(example + f" (update {i})")
        
        # Create DataFrame
        data = []
        for text in augmented_fake[:sample_size//2]:
            data.append({'text': text, 'label': 1})  # 1 = fake
        for text in augmented_real[:sample_size//2]:
            data.append({'text': text, 'label': 0})  # 0 = real
        
        return pd.DataFrame(data)
    
    def train_model(self, df=None, save_path=None):
        """Train the fake news detection model with enhanced features"""
        if df is None:
            df = self.create_sample_data()
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        X = df['processed_text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline with enhanced parameters
        self.pipeline = Pipeline([
            'tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 3),  # Include trigrams
                min_df=3,
                max_df=0.75,
                stop_words='english',
                sublinear_tf=True
            ),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight='balanced',
                max_depth=None,
                min_samples_split=5,
                n_jobs=-1
            ))
        ])
        
        # Train model with progress feedback
        print("Training model...")
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        return accuracy
    
    def predict(self, text, return_proba=False):
        """Predict if news is fake or real with enhanced error handling"""
        if self.pipeline is None:
            raise ValueError("Model not trained or loaded yet! Call train_model() or load_model() first.")
        
        if not text or not isinstance(text, str):
            return {'error': 'Invalid input text'}
        
        try:
            processed_text = self.preprocess_text(text)
            prediction = self.pipeline.predict([processed_text])[0]
            probability = self.pipeline.predict_proba([processed_text])[0]
            
            result = {
                'prediction': 'Fake' if prediction == 1 else 'Real',
                'confidence': float(max(probability)),
                'fake_probability': float(probability[1]),
                'real_probability': float(probability[0]),
                'text_processed': processed_text  # For debugging
            }
            
            if not return_proba:
                del result['fake_probability']
                del result['real_probability']
            
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def save_model(self, filepath='fake_news_model.joblib'):
        """Save trained model to file with version checking"""
        try:
            # Include metadata about versions
            metadata = {
                'numpy_version': np.__version__,
                'sklearn_version': sklearn.__version__,
                'pandas_version': pd.__version__,
                'save_timestamp': pd.Timestamp.now()
            }
            
            save_data = {
                'pipeline': self.pipeline,
                'metadata': metadata
            }
            
            joblib.dump(save_data, filepath)
            print(f"Model successfully saved to {filepath}")
            print(f"Saved with versions - NumPy: {metadata['numpy_version']}, "
                  f"scikit-learn: {metadata['sklearn_version']}")
        except Exception as e:
            print(f"Failed to save model: {str(e)}")
            raise
    
    def load_model(self, filepath='fake_news_model.joblib'):
        """Load trained model from file with version compatibility checks"""
        try:
            loaded_data = joblib.load(filepath)
            self.pipeline = loaded_data['pipeline']
            metadata = loaded_data.get('metadata', {})
            
            print(f"Model loaded from {filepath}")
            print(f"Original save versions - NumPy: {metadata.get('numpy_version', 'unknown')}, "
                  f"scikit-learn: {metadata.get('sklearn_version', 'unknown')}")
            print(f"Current versions - NumPy: {np.__version__}, "
                  f"scikit-learn: {sklearn.__version__}")
            
            # Version compatibility check
            current_sklearn = sklearn.__version__.split('.')[:2]
            saved_sklearn = metadata.get('sklearn_version', '0.0').split('.')[:2]
            
            if current_sklearn != saved_sklearn:
                print(f"Warning: Version mismatch! Model was saved with scikit-learn {metadata.get('sklearn_version')} "
                      f"but current version is {sklearn.__version__}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = FakeNewsDetector()
    
    # Option 1: Train new model
    print("Option 1: Training new model...")
    accuracy = detector.train_model(save_path='fake_news_model.joblib')
    
    # Option 2: Load existing model
    # print("Option 2: Loading existing model...")
    # detector.load_model('fake_news_model.joblib')
    
    # Test predictions
    test_articles = [
        "Breaking: Scientists discover cure for all diseases using this simple trick",
        "Local government approves new budget for public transportation improvements",
        "",  # Empty string test
        123,  # Non-string test
        "A new study published in Nature shows promising results for renewable energy"
    ]
    
    for article in test_articles:
        print(f"\nTesting article: {str(article)[:50]}...")
        try:
            result = detector.predict(article)
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")
                print(f"Processed text: {result.get('text_processed', '')[:100]}...")
        except Exception as e:
            print(f"Prediction failed: {str(e)}")