from flask import Flask, request, jsonify, send_file
import pickle
import numpy as np
import io
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

# --- START OF FIX ---
# This tells Matplotlib to use a non-interactive backend
# that doesn't try to create a GUI window.
import matplotlib
matplotlib.use('Agg') 
# --- END OF FIX ---

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

app = Flask(__name__)

# --- Database Configuration ---
DATABASE = 'users.db'

def get_db():
    """Connects to the SQLite database."""
    db = sqlite3.connect(DATABASE)
    # This allows us to access columns by name (e.g., user['phone_no'])
    db.row_factory = sqlite3.Row
    return db

def init_db():
    """Initializes the database and creates the users table if it doesn't exist."""
    if os.path.exists(DATABASE):
        return  # Database already exists

    print("Creating new database: users.db")
    try:
        with app.app_context():
            db = get_db()
            cursor = db.cursor()
            cursor.execute('''
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone_no TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                );
            ''')
            db.commit()
            db.close()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")

# --- Load Models ---
try:
    with open("score_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please make sure score_model.pkl, scaler.pkl, and feature_names.pkl are in the same directory.")
    
DISEASES = ["Alzheimer", "Parkinson", "Stress"]


# --- Helper Functions ---

def get_predictions(features):
    # Extract features in the correct order, defaulting to 0 if missing
    x = np.array([[features.get(f, 0) for f in feature_names]], dtype=np.float32)
    
    # Scale the features
    x_scaled = scaler.transform(x)
    
    # Get model predictions
    preds = model.predict(x_scaled)

    # Process predictions
    if preds.ndim > 1 and preds.shape[1] == len(DISEASES):
        scores = preds[0]
    else:
        # Fallback if model output is not as expected
        scores = [preds[0]] * len(DISEASES)  

    results = {}
    for i, disease in enumerate(DISEASES):
        score_prob = float(scores[i])
        score_percent = score_prob * 100 if score_prob <= 1 else score_prob
        
        stage = int(round(score_percent / 25)) 
        results[disease] = {"Risk_Score": round(score_percent, 2), "Risk_Stage": stage}
    return results

def generate_pdf_report(predictions):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        diseases = list(predictions.keys())
        scores = [predictions[d]["Risk_Score"] for d in diseases]
        stages = [predictions[d]["Risk_Stage"] for d in diseases]

        # Page 1: Risk Scores Bar Chart
        plt.figure(figsize=(8, 5))
        sns.barplot(x=diseases, y=scores, palette="coolwarm")
        plt.title("Predicted Disease Risk Scores (%)", fontsize=16)
        plt.ylabel("Risk Score (%)")
        plt.ylim(0, 100)
        for i, s in enumerate(scores):
            plt.text(i, s + 1, f"{s:.1f}%", ha='center')
        pdf.savefig()
        plt.close()

        # Page 2: Risk Stages Bar Chart
        plt.figure(figsize=(8, 5))
        sns.barplot(x=diseases, y=stages, palette="viridis")
        plt.title("Predicted Disease Stages", fontsize=16)
        plt.ylabel("Risk Stage")
        plt.yticks(range(0, max(stages) + 2))
        for i, st in enumerate(stages):
            plt.text(i, st + 0.1, f"Stage {st}", ha='center')
        pdf.savefig()
        plt.close()

        # Page 3: Summary Text
        plt.figure(figsize=(8, 5))
        plt.axis('off')
        summary = "\n".join([
            f"• {d}: {predictions[d]['Risk_Score']}% Risk (Stage {predictions[d]['Risk_Stage']})"
            for d in diseases
        ])
        report_text = f"""
        NeuroPredict Diagnostic Report
        
        Summary of Findings:
        {summary}
        
        
        Disclaimer:
        This is an AI-generated insight based on the provided data.
        It is not a medical diagnosis. Please consult a qualified
        healthcare professional for any medical advice.
        """
        plt.text(0.05, 0.9, report_text, fontsize=12, va='top', ha='left', family='monospace')
        pdf.savefig()
        plt.close()

    buffer.seek(0)
    return buffer
import requests

def generate_personalized_care_plan(predictions):
    """
    Takes model predictions dict and generates a clinical lifestyle plan.
    Compatible with get_predictions() format.
    """
    
    api_key = os.getenv("GROQ_API_KEY")

    summary_lines = []
    for disease, info in predictions.items():
        score = info.get("Risk_Score", "N/A")
        stage = info.get("Risk_Stage", "N/A")
        summary_lines.append(f"{disease}: Risk Score = {score}%, Stage = {stage}")
    
    report_text = "\n".join(summary_lines)

    system_prompt = """
You are a medical lifestyle guidance assistant.
You DO NOT diagnose disease.
You provide clinically safe evidence-based recommendations.
Tone: Formal, medical, structured.
"""

    user_prompt = f"""
Patient Neurological Risk Summary:
{report_text}

Generate a structured clinical care plan containing:

1. Interpretation Summary
2. Diet & Nutrition Guidance
3. Physical Exercise Plan (Weekly Routine)
4. Cognitive & Brain Health Activities
5. Stress Management & Breathing Exercises
6. Sleep & Recovery Routine
7. Monitoring & When to Seek Medical Evaluation
"""

    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": "llama-3.1-8b-instant", 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.35
    }

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json=payload
    )

    if response.status_code != 200:
        print("Groq API Error:", response.json())
        return None

    return response.json()["choices"][0]["message"]["content"]



# --- API ROUTES ---

@app.route("/")
def home():
    return jsonify({"message": "NeuroPredict API is running. Use /login, /predict, or /report."})

@app.route("/login", methods=["GET","POST"])
def login():
    """
    Handles user login. If the user doesn't exist, it creates a new user.
    """
    data = request.get_json()
    if not data or 'phone_no' not in data or 'password' not in data:
        return jsonify({"error": "phone_no and password are required."}), 400

    phone_no = data['phone_no']
    password = data['password']

    # Simple validation for phone number
    if not phone_no or not phone_no.isdigit() or len(phone_no) != 10:
        return jsonify({"error": "Invalid phone no: must be 10 digits."}), 400

    db = get_db()
    cursor = db.cursor()

    try:
        # Check if user exists
        cursor.execute("SELECT * FROM users WHERE phone_no = ?", (phone_no,))
        user = cursor.fetchone()

        if user:
            # --- USER EXISTS ---
            # Check if the provided password matches the stored hash
            if check_password_hash(user['password_hash'], password):
                db.close()
                return jsonify({"message": f"Login successful for {phone_no}."}), 200
            else:
                db.close()
                return jsonify({"error": "Invalid password."}), 401
        else:
            # --- USER DOES NOT EXIST ---
            # Create a new user
            hashed_password = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO users (phone_no, password_hash) VALUES (?, ?)",
                (phone_no, hashed_password)
            )
            db.commit()
            db.close()
            # 201 Created
            return jsonify({"message": f"User {phone_no} created successfully."}), 201

    except sqlite3.IntegrityError:
        # This handles a rare case where two requests try to create the same user
        db.close()
        return jsonify({"error": "User already exists. Please try logging in."}), 409
    except Exception as e:
        db.close()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route("/predict", methods=["GET","POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided."}), 400
    try:
        predictions = get_predictions(data)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route("/report", methods=["GET","POST"])
def report():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided."}), 400
    try:
        predictions = get_predictions(data)
        pdf_buf = generate_pdf_report(predictions)
        return send_file(
            pdf_buf, 
            as_attachment=True, 
            download_name="NeuroPredict_Report.pdf", 
            mimetype="application/pdf"
        )
    except Exception as e:
        return jsonify({"error": f"Report generation error: {str(e)}"}), 500
@app.route("/recommend", methods=["GET","POST"])
def recommend():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided."}), 400
    try:
        predictions = get_predictions(data)
        care_plan = generate_personalized_care_plan(predictions)

        return jsonify({
            "predictions": predictions,
            "personalized_care_plan": care_plan
        })

    except Exception as e:
        return jsonify({"error": f"Recommendation generation error: {str(e)}"}), 500


if __name__ == "__main__":
    # Initialize the database (creates users.db if it doesn't exist)
    init_db()
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=3000, debug=True)