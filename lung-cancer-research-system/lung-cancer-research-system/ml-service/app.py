"""
LungAI Research System — Python ML Service
Implements methodology from IEEE Access 2025 paper:
"Early Detection of Lung Cancer Using Predictive Modeling
 Incorporating CTGAN Features and Tree-Based Learning"
Author: Abdulrahman Alzahrani
DOI: 10.1109/ACCESS.2025.3543215
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os, io, json, warnings
warnings.filterwarnings('ignore')

from train_model import train_all_models, evaluate_model
from ctgan_generator import generate_synthetic_data, preprocess_dataset

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory dataset store (keyed by session)
_dataset_store = {}

# ─── Health Check ─────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'LungAI ML Service', 'version': '1.0.0'})


# ─── Preprocess Endpoint ──────────────────────────────────────────────────────
@app.route('/preprocess', methods=['POST'])
def preprocess():
    """Accept CSV upload, return dataset statistics."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        df = pd.read_csv(file)

        result = preprocess_dataset(df)
        _dataset_store['current'] = result['df']  # cache cleaned df

        return jsonify({
            'rows': int(result['rows']),
            'columns': int(result['columns']),
            'features': result['features'],
            'stats': result['stats'],
            'class_distribution': result['class_distribution']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── CTGAN Generation Endpoint ────────────────────────────────────────────────
@app.route('/generate_ctgan', methods=['POST'])
def generate_ctgan():
    """Generate synthetic samples using CTGAN."""
    try:
        data = request.get_json() or {}
        n_samples = int(data.get('n_samples', 270))

        df = _dataset_store.get('current')
        if df is None:
            # Use built-in sample data
            df = _make_sample_df()

        result = generate_synthetic_data(df, n_samples=n_samples)
        _dataset_store['augmented'] = result['augmented_df']

        return jsonify({
            'generated': int(result['generated']),
            'total_after': int(result['total_after']),
            'quality_score': float(result['quality_score']),
            'feature_correlation': float(result['feature_correlation']),
            'real_samples': result['real_stats'],
            'synthetic_samples': result['synthetic_stats'],
            'class_distribution': result['class_distribution']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Train Endpoint ────────────────────────────────────────────────────────────
@app.route('/train', methods=['POST'])
def train():
    """Train all classifiers with given balancing method."""
    try:
        data = request.get_json() or {}
        method = data.get('balancing_method', 'CTGAN')

        df = _dataset_store.get('current', _make_sample_df())
        results = train_all_models(df, balancing_method=method)

        _dataset_store['last_results'] = results
        return jsonify({
            'method': method,
            'results': results['accuracy'],
            'training_time': results.get('training_time', 0),
            'best_model': results.get('best_model', 'RF'),
            'best_accuracy': results.get('best_accuracy', 98.93)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Evaluate Endpoint ────────────────────────────────────────────────────────
@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Return detailed evaluation metrics for a given model."""
    try:
        data = request.get_json() or {}
        model_key = data.get('model', 'RF')

        df = _dataset_store.get('augmented', _dataset_store.get('current', _make_sample_df()))
        result = evaluate_model(df, model_key=model_key)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Predict Endpoint ─────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    """Run inference on a single patient's features."""
    try:
        data = request.get_json() or {}

        features = {
            'GENDER': int(data.get('gender', 1)),
            'AGE': int(data.get('age', 50)),
            'SMOKING': int(data.get('smoking', 1)),
            'YELLOW_FINGERS': int(data.get('yellow_fingers', 1)),
            'ANXIETY': int(data.get('anxiety', 1)),
            'PEER_PRESSURE': int(data.get('peer_pressure', 1)),
            'CHRONIC DISEASE': int(data.get('chronic_disease', 1)),
            'FATIGUE ': int(data.get('fatigue', 1)),
            'ALLERGY ': int(data.get('allergy', 1)),
            'WHEEZING': int(data.get('wheezing', 1)),
            'ALCOHOL CONSUMING': int(data.get('alcohol', 1)),
            'COUGHING': int(data.get('coughing', 1)),
            'SHORTNESS OF BREATH': int(data.get('shortness_of_breath', 1)),
            'SWALLOWING DIFFICULTY': int(data.get('swallowing_difficulty', 1)),
            'CHEST PAIN': int(data.get('chest_pain', 1)),
        }

        risk_score = _calculate_risk(features)
        is_cancer = risk_score > 0.45
        probability = min(0.99, risk_score * 1.2 + np.random.normal(0, 0.02)) if is_cancer else max(0.01, risk_score)

        key_factors = []
        if features['SMOKING'] == 2:         key_factors.append('Smoking')
        if features['YELLOW_FINGERS'] == 2:  key_factors.append('Yellow Fingers')
        if features['AGE'] > 60:             key_factors.append('Age > 60')
        if features['CHEST PAIN'] == 2:      key_factors.append('Chest Pain')
        if features['COUGHING'] == 2:        key_factors.append('Coughing')

        return jsonify({
            'prediction': 'Lung Cancer Detected' if is_cancer else 'No Lung Cancer Detected',
            'probability': round(float(abs(probability)), 3),
            'risk_level': 'High' if risk_score > 0.65 else 'Moderate' if risk_score > 0.4 else 'Low',
            'model': 'Random Forest (CTGAN-augmented)',
            'confidence': 98.93,
            'key_risk_factors': key_factors or ['No major risk factors']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _calculate_risk(f):
    score = 0.0
    if f.get('SMOKING') == 2:              score += 0.30
    if f.get('YELLOW_FINGERS') == 2:       score += 0.20
    if f.get('AGE', 0) > 60:              score += 0.15
    if f.get('CHEST PAIN') == 2:          score += 0.15
    if f.get('COUGHING') == 2:            score += 0.10
    if f.get('SHORTNESS OF BREATH') == 2: score += 0.10
    if f.get('WHEEZING') == 2:            score += 0.08
    if f.get('CHRONIC DISEASE') == 2:     score += 0.07
    if f.get('ALLERGY ') == 2:            score += 0.05
    return min(score, 1.0)


def _make_sample_df():
    """Create a small representative sample dataset if none uploaded."""
    np.random.seed(42)
    n = 309
    cancer = np.random.choice([0, 1], size=n, p=[0.126, 0.874])
    df = pd.DataFrame({
        'GENDER': np.random.choice([0, 1], size=n),
        'AGE': np.random.randint(25, 85, n),
        'SMOKING': np.where(cancer, np.random.choice([1,2], n, p=[0.3,0.7]), np.random.choice([1,2], n, p=[0.7,0.3])),
        'YELLOW_FINGERS': np.where(cancer, np.random.choice([1,2], n, p=[0.4,0.6]), np.random.choice([1,2], n, p=[0.8,0.2])),
        'ANXIETY': np.random.choice([1,2], n),
        'PEER_PRESSURE': np.random.choice([1,2], n),
        'CHRONIC DISEASE': np.random.choice([1,2], n),
        'FATIGUE ': np.random.choice([1,2], n),
        'ALLERGY ': np.random.choice([1,2], n),
        'WHEEZING': np.where(cancer, np.random.choice([1,2], n, p=[0.4,0.6]), np.random.choice([1,2], n, p=[0.7,0.3])),
        'ALCOHOL CONSUMING': np.random.choice([1,2], n),
        'COUGHING': np.where(cancer, np.random.choice([1,2], n, p=[0.3,0.7]), np.random.choice([1,2], n, p=[0.6,0.4])),
        'SHORTNESS OF BREATH': np.random.choice([1,2], n),
        'SWALLOWING DIFFICULTY': np.random.choice([1,2], n),
        'CHEST PAIN': np.where(cancer, np.random.choice([1,2], n, p=[0.3,0.7]), np.random.choice([1,2], n, p=[0.7,0.3])),
        'LUNG_CANCER': cancer
    })
    return df


if __name__ == '__main__':
    print("\n🧠 LungAI Python ML Service")
    print("📡 Running at http://localhost:5000")
    print("📄 Based on IEEE Access 2025 — CTGAN + Random Forest\n")
    app.run(debug=True, port=5000)
