import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Path to your pickled model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "decision_model.pkl")

# Load model safely
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# Features expected by decision_model.pkl
FEATURE_NAMES = ["Age", "Gender", "Region", "Occupation", "Income"]

# Categorical mapping options (Adjust labels if your preprocessor uses specific encodings)
CATEGORICAL_OPTIONS = {
    "Gender": ["Male", "Female", "Other"],
    "Region": ["Urban", "Suburban", "Rural"],
    "Occupation": ["Professional", "Management", "Skilled", "Student", "Unemployed"]
}

# Embedded HTML Template with Glassmorphic modern dark styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Decision Predictor App</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-glow: rgba(99, 102, 241, 0.25);
            --accent-primary: #6366f1;
            --accent-hover: #4f46e5;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 650px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 24px;
            border: 1px solid var(--border-glow);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            padding: 2.5rem;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 1.85rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.25rem;
            margin-bottom: 2rem;
        }

        @media (min-width: 500px) {
            .form-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .full-width {
                grid-column: span 2;
            }
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
            letter-spacing: 0.02em;
        }

        .input-group input, .input-group select {
            width: 100%;
            padding: 0.85rem 1rem;
            background-color: var(--input-bg);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            color: var(--text-primary);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
        }

        .submit-btn {
            width: 100%;
            padding: 1rem;
            background-color: var(--accent-primary);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39);
        }

        .submit-btn:hover {
            background-color: var(--accent-hover);
            transform: translateY(-1px);
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            animation: fadeIn 0.3s ease-in-out;
        }

        .result-card h3 {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 0.3rem;
        }

        .result-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: #818cf8;
            text-transform: capitalize;
        }

        .error-card {
            margin-top: 2rem;
            padding: 1.25rem;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #fca5a5;
            border-radius: 12px;
            text-align: center;
            font-size: 0.9rem;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Decision Classifier</h1>
            <p>Fill in the criteria below to generate a model prediction</p>
        </div>

        <form action="/predict" method="POST">
            <div class="form-grid">
                {% for feature in features %}
                <div class="input-group {% if loop.last and loop.index % 2 != 0 %}full-width{% endif %}">
                    <label for="{{ feature }}">{{ feature }}</label>
                    {% if feature in categorical_opts %}
                        <select name="{{ feature }}" id="{{ feature }}" required>
                            {% for option in categorical_opts[feature] %}
                                <option value="{{ option }}" {% if form_data and form_data.get(feature) == option %}selected{% endif %}>
                                    {{ option }}
                                </option>
                            {% endfor %}
                        </select>
                    {% else %}
                        <input type="number" step="any" name="{{ feature }}" id="{{ feature }}" 
                               placeholder="Enter {{ feature }}" 
                               value="{{ form_data.get(feature, '') if form_data else '' }}" required>
                    {% endif %}
                </div>
                {% endfor %}
            </div>

            <button type="submit" class="submit-btn">Run Prediction</button>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            <h3>Prediction Output</h3>
            <div class="value">{{ prediction }}</div>
        </div>
        {% endif %}

        {% if error %}
        <div class="error-card">
            {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML_TEMPLATE,
        features=FEATURE_NAMES,
        categorical_opts=CATEGORICAL_OPTIONS,
        prediction=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE,
            features=FEATURE_NAMES,
            categorical_opts=CATEGORICAL_OPTIONS,
            error="Error: 'decision_model.pkl' could not be loaded on the server.",
            prediction=None
        )

    try:
        raw_data = request.form.to_dict()
        input_values = []

        for feature in FEATURE_NAMES:
            val = raw_data.get(feature)
            # Numeric conversion logic
            if feature in CATEGORICAL_OPTIONS:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    # Deterministic hash encoding fallback if model expects numbers for category strings
                    val = abs(hash(val)) % 100
            else:
                val = float(val)
            input_values.append(val)

        # Convert to DataFrame / Numpy array as required by Scikit-Learn
        df_input = pd.DataFrame([input_values], columns=FEATURE_NAMES)
        prediction_val = model.predict(df_input)[0]

        return render_template_string(
            HTML_TEMPLATE,
            features=FEATURE_NAMES,
            categorical_opts=CATEGORICAL_OPTIONS,
            prediction=str(prediction_val),
            form_data=raw_data
        )

    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            features=FEATURE_NAMES,
            categorical_opts=CATEGORICAL_OPTIONS,
            error=f"Prediction Error: {str(e)}",
            form_data=request.form.to_dict(),
            prediction=None
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
