import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the pickle decision tree model
MODEL_PATH = "decision_model_.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


# HTML, CSS (with Animations), and JS in a single template string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Decision Predictor</title>
    <!-- Animate.css for smooth entrance animations -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"/>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            width: 100%;
            max-width: 500px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        h2 {
            color: #333;
            text-align: center;
            margin-bottom: 25px;
            font-weight: 700;
        }

        .form-group {
            margin-bottom: 20px;
            position: relative;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 600;
            font-size: 0.9rem;
        }

        input, select {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1rem;
            transition: all 0.3s ease;
            outline: none;
            background: #f9f9f9;
        }

        input:focus, select:focus {
            border-color: #667eea;
            background: #fff;
            box-shadow: 0 0 8px rgba(102, 126, 234, 0.3);
        }

        .btn-submit {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(118, 75, 162, 0.4);
        }

        .btn-submit:hover {
            opacity: 0.95;
            transform: scale(1.02);
            box-shadow: 0 8px 20px rgba(118, 75, 162, 0.6);
        }

        .btn-submit:active {
            transform: scale(0.98);
        }

        .result-box {
            margin-top: 25px;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 700;
        }

        .result-yes {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .result-no {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }

        /* Pulse glow for prediction result */
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7); }
            70% { box-shadow: 0 0 0 15px rgba(102, 126, 234, 0); }
            100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
        }

        .pulse {
            animation: pulseGlow 1.5s infinite;
        }
    </style>
</head>
<body>

<div class="card animate__animated animate__fadeInDown">
    <h2>🎯 Model Predictor</h2>

    <form action="/predict" method="POST">
        <div class="form-group">
            <label for="age">Age</label>
            <input type="number" id="age" name="age" placeholder="e.g. 25" required value="{{ form_data.get('age', '') }}">
        </div>

        <div class="form-group">
            <label for="gender">Gender</label>
            <select id="gender" name="gender" required>
                <option value="" disabled {% if not form_data.get('gender') %}selected{% endif %}>Select Gender</option>
                <option value="0" {% if form_data.get('gender') == '0' %}selected{% endif %}>Female (0)</option>
                <option value="1" {% if form_data.get('gender') == '1' %}selected{% endif %}>Male (1)</option>
            </select>
        </div>

        <div class="form-group">
            <label for="region">Region Code</label>
            <input type="number" id="region" name="region" placeholder="e.g. 0, 1, 2..." required value="{{ form_data.get('region', '') }}">
        </div>

        <div class="form-group">
            <label for="occupation">Occupation Code</label>
            <input type="number" id="occupation" name="occupation" placeholder="e.g. 0, 1, 2..." required value="{{ form_data.get('occupation', '') }}">
        </div>

        <div class="form-group">
            <label for="income">Income</label>
            <input type="number" step="any" id="income" name="income" placeholder="e.g. 50000" required value="{{ form_data.get('income', '') }}">
        </div>

        <button type="submit" class="btn-submit">Predict Outcome</button>
    </form>

    {% if prediction %}
        <div class="result-box animate__animated animate__zoomIn pulse {% if prediction == 'yes' %}result-yes{% else %}result-no{% endif %}">
            Prediction Result: {{ prediction.upper() }}
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, prediction=None, form_data={})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model not loaded properly. Ensure decision_model_.pkl is in the project root.", 500

    try:
        # Get raw form data
        age = float(request.form.get("age"))
        gender = float(request.form.get("gender"))
        region = float(request.form.get("region"))
        occupation = float(request.form.get("occupation"))
        income = float(request.form.get("income"))

        # Features order according to model: ['Age', 'Gender', 'Region', 'Occupation', 'Income']
        input_data = np.array([[age, gender, region, occupation, income]])
        
        # Predict using loaded pickle decision tree model
        prediction = model.predict(input_data)[0]

        return render_template_string(
            HTML_TEMPLATE, 
            prediction=str(prediction), 
            form_data=request.form
        )

    except Exception as e:
        return f"An error occurred during prediction: {str(e)}", 400

if __name__ == "__main__":
    app.run(debug=True)
