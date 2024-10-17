from flask import Blueprint, request, jsonify
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from .new_data import preprocess_new_data,VOC
from .preprocesing_email import prepare

main = Blueprint('main', __name__)

# Load the models when the application starts
with open('./links_classify.pkl', 'rb') as f:
    links_model = pickle.load(f)

phishing_model = load_model('./phishing_email_detection.h5')

@main.route('/analyze', methods=['POST'])
def analyze_email():
    data = request.json
    email_content = data.get('email', '')
    links = data.get('links', [])

    # Prepare data for links classification
    if len(links) > 0:
        links_df = pd.DataFrame(links, columns=['url']) 
        links_df = preprocess_new_data(links_df,VOC)
        links_predictions = links_model.predict(links_df)
        links_analysis = ['suspicious' if pred == 1 else 'safe' for pred in links_predictions]
    else:
        links_analysis = 'No links identified!'

    # Prepare data for phishing detection
    email_input = pd.Series([email_content])
    email_input = prepare(email_input) 
    phishing_prediction = phishing_model.predict(email_input)

    is_suspicious_email = phishing_prediction[0] == 1  
    analysis_result = {
        'isSuspicious': is_suspicious_email,
        'linksAnalysis': links_analysis,
        'analysis': 'This email contains suspicious content.' if is_suspicious_email else 'Email appears safe.',
    }
    
    # Convert any ndarray values in analysis_result to lists if necessary
    for key in analysis_result:
        if isinstance(analysis_result[key], np.ndarray):
            analysis_result[key] = analysis_result[key].tolist()

    return jsonify(analysis_result)
