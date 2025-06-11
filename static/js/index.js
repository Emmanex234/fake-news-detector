function setExample(element) {
            document.getElementById('newsText').value = element.textContent.replace(/"/g, '');
        }

        function clearText() {
            document.getElementById('newsText').value = '';
            document.getElementById('resultSection').style.display = 'none';
            document.getElementById('errorMessage').style.display = 'none';
        }

        function showError(message) {
            const errorDiv = document.getElementById('errorMessage');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }

        function hideError() {
            document.getElementById('errorMessage').style.display = 'none';
        }

async function analyzeNews() {
    const text = document.getElementById('newsText').value.trim();
    const loading = document.getElementById('loading');
    const resultSection = document.getElementById('resultSection');
    const errorDiv = document.getElementById('errorMessage');

    // Clear previous results
    errorDiv.style.display = 'none';
    resultSection.style.display = 'none';
    loading.style.display = 'block';

    try {
        if (!text) {
            throw new Error('Please enter some text to analyze');
        }

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Server error');
        }

        // Display results
        displayResult(result);

    } catch (error) {
        errorDiv.textContent = error.message;
        errorDiv.style.display = 'block';
        console.error('Error:', error);
    } finally {
        loading.style.display = 'none';
    }
}

function displayResult(result) {
    const resultSection = document.getElementById('resultSection');
    const isFake = result.prediction === 'Fake';

    // Update UI
    resultSection.className = `result-section ${isFake ? 'result-fake' : 'result-real'}`;
    document.getElementById('resultIcon').textContent = isFake ? '⚠️' : '✅';
    document.getElementById('resultTitle').textContent = 
        isFake ? 'Likely FAKE News' : 'Likely REAL News';
    
    // Update confidence meter
    const confidenceFill = document.getElementById('confidenceFill');
    confidenceFill.style.width = `${result.confidence * 100}%`;
    confidenceFill.className = `confidence-fill ${isFake ? 'confidence-fake' : 'confidence-real'}`;
    
    // Update percentages
    document.getElementById('confidenceValue').textContent = `${(result.confidence * 100).toFixed(1)}%`;
    document.getElementById('fakeProb').textContent = `${(result.fake_probability * 100).toFixed(1)}%`;
    document.getElementById('realProb').textContent = `${(result.real_probability * 100).toFixed(1)}%`;
    
    // Show result
    resultSection.style.display = 'block';
}
function displayResult(result) {
    try {
        const resultSection = document.getElementById('resultSection');
        const isFake = result.prediction === 'Fake';
        
        // Set result class and content
        resultSection.className = 'result-section ' + (isFake ? 'result-fake' : 'result-real');
        
        document.getElementById('resultIcon').textContent = isFake ? '⚠️' : '✅';
        document.getElementById('resultTitle').textContent = 
            isFake ? 'Likely FAKE News' : 'Likely REAL News';
        
        // Set confidence bar
        const confidenceFill = document.getElementById('confidenceFill');
        confidenceFill.className = 'confidence-fill ' + (isFake ? 'confidence-fake' : 'confidence-real');
        confidenceFill.style.width = (result.confidence * 100) + '%';
        
        // Set statistics - with fallbacks in case probabilities aren't returned
        document.getElementById('confidenceValue').textContent = (result.confidence * 100).toFixed(1) + '%';
        document.getElementById('realProb').textContent = 
            (result.real_probability !== undefined ? (result.real_probability * 100).toFixed(1) : 'N/A') + '%';
        document.getElementById('fakeProb').textContent = 
            (result.fake_probability !== undefined ? (result.fake_probability * 100).toFixed(1) : 'N/A') + '%';
        
        // Show result
        resultSection.style.display = 'block';
    } catch (error) {
        console.error('Error displaying result:', error);
        showError('Error displaying results: ' + error.message);
    }
}
        function displayResult(result) {
            const resultSection = document.getElementById('resultSection');
            const isFake = result.prediction === 'Fake';
            
            // Set result class and content
            resultSection.className = 'result-section ' + (isFake ? 'result-fake' : 'result-real');
            
            document.getElementById('resultIcon').textContent = isFake ? '⚠️' : '✅';
            document.getElementById('resultTitle').textContent = 
                isFake ? 'Likely FAKE News' : 'Likely REAL News';
            
            // Set confidence bar
            const confidenceFill = document.getElementById('confidenceFill');
            confidenceFill.className = 'confidence-fill ' + (isFake ? 'confidence-fake' : 'confidence-real');
            confidenceFill.style.width = (result.confidence * 100) + '%';
            
            // Set statistics
            document.getElementById('confidenceValue').textContent = (result.confidence * 100).toFixed(1) + '%';
            document.getElementById('realProb').textContent = (result.real_probability * 100).toFixed(1) + '%';
            document.getElementById('fakeProb').textContent = (result.fake_probability * 100).toFixed(1) + '%';
            
            // Show result
            resultSection.style.display = 'block';
        }

        // Handle Enter key in textarea (Ctrl+Enter to analyze)
        document.getElementById('newsText').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                analyzeNews();
            }
        });