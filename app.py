from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from fake_news_detector import FakeNewsDetector
import os
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS

# Initialize detector
detector = FakeNewsDetector()
MODEL_PATH = 'fake_news_model.joblib'

def initialize_model():
    """Load or train the model on startup"""
    try:
        if os.path.exists(MODEL_PATH):
            logger.info("Loading pre-trained model...")
            detector.load_model(MODEL_PATH)
        else:
            logger.info("No model found. Training new model...")
            df = detector.create_sample_data()
            detector.train_model(df)
            detector.save_model(MODEL_PATH)
        logger.info("Model ready")
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Make prediction
        result = detector.predict(text)
        
        # Validate output
        if not all(key in result for key in ['prediction', 'confidence']):
            raise ValueError("Invalid prediction format")
        
        return jsonify({
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'fake_probability': result.get('fake_probability', 0),
            'real_probability': result.get('real_probability', 0),
            'processed_text': result.get('text_processed', '')[:100] + '...'
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

# Initialize the model when starting the app
initialize_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)