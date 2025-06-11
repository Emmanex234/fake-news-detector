# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import os
from fake_news_detector import FakeNewsDetector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global model variable
detector = None

def load_model():
    """Load the trained model on startup"""
    global detector
    try:
        detector = FakeNewsDetector()
        if os.path.exists('fake_news_model.pkl'):
            detector.load_model('fake_news_model.pkl')
            logger.info("Model loaded successfully")
        else:
            # Train model if no saved model exists
            logger.info("No saved model found. Training new model...")
            df = detector.create_sample_data()
            detector.train_model(df)
            detector.save_model()
            logger.info("New model trained and saved")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        detector = None

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if news article is fake or real"""
    try:
        if detector is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Make prediction
        result = detector.predict(text)
        
        # Add input text to response
        result['input_text'] = text[:200] + ('...' if len(text) > 200 else '')
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict-url', methods=['POST'])
def predict_url():
    """Predict from URL (placeholder for future implementation)"""
    return jsonify({
        'error': 'URL prediction not implemented yet. Please paste the article text directly.'
    }), 501

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Predict multiple articles at once"""
    try:
        if detector is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        if not data or 'articles' not in data:
            return jsonify({'error': 'Missing articles field'}), 400
        
        articles = data['articles']
        
        if not isinstance(articles, list):
            return jsonify({'error': 'Articles must be a list'}), 400
        
        if len(articles) > 10:  # Limit batch size
            return jsonify({'error': 'Maximum 10 articles per batch'}), 400
        
        results = []
        for i, text in enumerate(articles):
            if not text.strip():
                results.append({'error': f'Article {i+1} is empty'})
                continue
            
            try:
                result = detector.predict(text)
                result['article_index'] = i + 1
                result['input_text'] = text[:100] + ('...' if len(text) > 100 else '')
                results.append(result)
            except Exception as e:
                results.append({'error': f'Error processing article {i+1}: {str(e)}'})
        
        return jsonify({'results': results})
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    if detector is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'Random Forest with TF-IDF',
        'features': 'Text content analysis',
        'classes': ['Real News', 'Fake News'],
        'status': 'ready'
    })

# Initialize model on startup
load_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)