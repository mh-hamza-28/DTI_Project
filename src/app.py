import os
import sys

# Ensure src is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from features import drug_features, protein_features
import pandas as pd
from model import prepare_data, train_and_eval

app = Flask(__name__, template_folder='../templates', static_folder='../static')
# Enable cross-origin requests so your GitHub pages frontend can talk to this backend
CORS(app)

lr_model = None
rf_model = None

def init_app():
    global lr_model, rf_model
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "dti_data.csv")
    try:
        df = pd.read_csv(data_path)
        X, y = prepare_data(df)
        lr_model, rf_model = train_and_eval(X, y)
    except Exception as e:
        print(f"Failed to initialize models on startup: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    global lr_model, rf_model
    if lr_model is None:
        init_app()
        if lr_model is None:
            return jsonify({"error": "Failed to train models. Data might be missing."}), 500
            
    data = request.json
    smiles = data.get("smiles", "")
    sequence = data.get("sequence", "")
    
    d_feat = drug_features(smiles)
    p_feat = protein_features(sequence)
    
    if d_feat is None:
        return jsonify({"error": "Invalid SMILES structure."}), 400
        
    if p_feat is None:
        return jsonify({"error": "Invalid protein sequence."}), 400
        
    features = [list(d_feat) + list(p_feat)]
    prob_lr = lr_model.predict_proba(features)[0][1]
    prob_rf = rf_model.predict_proba(features)[0][1]
    
    avg_score = (prob_lr + prob_rf) / 2
    prediction_label = "Interacts" if avg_score > 0.5 else "Does Not Interact"
    
    return jsonify({
        "interaction": prediction_label,
        "confidence": float(avg_score * 100),
        "lr_score": float(prob_lr),
        "rf_score": float(prob_rf)
    })

if __name__ == "__main__":
    init_app()
    app.run(debug=True, port=5000)
