import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load Decision Tree Model
MODEL_PATH = "decision_model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None

model = load_model()

# Model Expects Features: ['Age', 'Gender', 'Region', 'Occupation', 'Income']
# Feature mappings for categorical inputs
GENDER_MAP = {"Male": 1, "Female": 0, "Other": 2}
REGION_MAP = {"Urban": 1, "Semi-Urban": 2, "Rural": 0}
OCCUPATION_MAP = {"Student": 0, "Salaried": 1, "Self-Employed": 2, "Other": 3}

HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en" data-theme="cyberpunk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cap Round Institute Prediction</title>
    
    <!-- Google Fonts & Libraries -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root[data-theme="cyberpunk"] {
            --bg-primary: #0a0e17;
            --bg-secondary: #131b2e;
            --card-bg: rgba(23, 32, 54, 0.7);
            --accent-primary: #00f2fe;
            --accent-secondary: #4facfe;
            --accent-gradient: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            --accent-pink: #f72585;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: rgba(0, 242, 254, 0.2);
            --shadow: 0 8px 32px 0 rgba(0, 242, 254, 0.15);
        }

        :root[data-theme="corporate"] {
            --bg-primary: #f1f5f9;
            --bg-secondary: #ffffff;
            --card-bg: rgba(255, 255, 255, 0.8);
            --accent-primary: #2563eb;
            --accent-secondary: #3b82f6;
            --accent-gradient: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            --accent-pink: #7c3aed;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border-color: rgba(37, 99, 235, 0.2);
            --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
        }

        :root[data-theme="sunset"] {
            --bg-primary: #180e29;
            --bg-secondary: #28153d;
            --card-bg: rgba(40, 21, 61, 0.7);
            --accent-primary: #ff758c;
            --accent-secondary: #ff7eb3;
            --accent-gradient: linear-gradient(135deg, #ff758c 0%, #ff7eb3 100%);
            --accent-pink: #7209b7;
            --text-main: #ffffff;
            --text-muted: #b8a7cb;
            --border-color: rgba(255, 117, 140, 0.25);
            --shadow: 0 8px 32px 0 rgba(255, 117, 140, 0.2);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem;
            overflow-x: hidden;
        }

        h1, h2, h3, .brand {
            font-family: 'Space Grotesk', sans-serif;
        }

        /* App Layout */
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 2rem;
        }

        /* Glassmorphism Header */
        header {
            grid-column: 1 / -1;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.5rem 2rem;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            box-shadow: var(--shadow);
            margin-bottom: 0.5rem;
        }

        .brand-title {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.6rem;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Theme Selector */
        .theme-selector {
            display: flex;
            gap: 0.5rem;
            background: var(--bg-secondary);
            padding: 4px;
            border-radius: 30px;
            border: 1px solid var(--border-color);
        }

        .theme-btn {
            border: none;
            background: transparent;
            color: var(--text-muted);
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.85rem;
        }

        .theme-btn.active {
            background: var(--accent-gradient);
            color: #fff;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }

        /* Form Card */
        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.8rem;
            box-shadow: var(--shadow);
        }

        .form-group {
            margin-bottom: 1.2rem;
        }

        .form-group label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .input-control {
            width: 100%;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
        }

        .input-control:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(0, 242, 254, 0.15);
        }

        .btn-predict {
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 12px;
            background: var(--accent-gradient);
            color: #fff;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 1rem;
            box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(0, 242, 254, 0.4); }
            70% { box-shadow: 0 0 0 12px rgba(0, 242, 254, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 242, 254, 0); }
        }

        /* Visualizations Grid */
        .analytics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }

        .full-width {
            grid-column: span 2;
        }

        .result-banner {
            text-align: center;
            padding: 1.5rem;
            border-radius: 16px;
            background: var(--bg-secondary);
            border: 2px dashed var(--accent-primary);
            margin-bottom: 1.5rem;
        }

        .status-badge {
            display: inline-block;
            font-size: 2rem;
            font-weight: 800;
            margin-top: 8px;
            padding: 4px 20px;
            border-radius: 30px;
        }

        .status-yes {
            color: #10b981;
            background: rgba(16, 185, 129, 0.15);
        }

        .status-no {
            color: #ef4444;
            background: rgba(239, 68, 68, 0.15);
        }

        .chart-container {
            position: relative;
            height: 250px;
            width: 100%;
        }

        /* Responsive */
        @media (max-width: 1024px) {
            .dashboard-container {
                grid-template-columns: 1fr;
            }
            .analytics-grid {
                grid-template-columns: 1fr;
            }
            .full-width {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>

    <div class="dashboard-container">
        <!-- Header Section -->
        <header>
            <div class="brand-title">
                <i class="fa-solid fa-graduation-cap"></i>
                <span>Cap Round Institute Prediction</span>
            </div>
            <div class="theme-selector">
                <button class="theme-btn active" onclick="setTheme('cyberpunk')">Cyberpunk</button>
                <button class="theme-btn" onclick="setTheme('corporate')">Corporate</button>
                <button class="theme-btn" onclick="setTheme('sunset')">Sunset</button>
            </div>
        </header>

        <!-- Form Control Section -->
        <div class="glass-card">
            <h3 style="margin-bottom: 1.5rem; color: var(--accent-primary);">
                <i class="fa-solid fa-sliders"></i> Candidate Parameters
            </h3>
            
            <form id="predictionForm">
                <div class="form-group">
                    <label>Age</label>
                    <input type="number" id="age" name="age" class="input-control" value="22" min="15" max="100" required>
                </div>

                <div class="form-group">
                    <label>Gender</label>
                    <select id="gender" name="gender" class="input-control">
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Region</label>
                    <select id="region" name="region" class="input-control">
                        <option value="Urban">Urban</option>
                        <option value="Semi-Urban">Semi-Urban</option>
                        <option value="Rural">Rural</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Occupation</label>
                    <select id="occupation" name="occupation" class="input-control">
                        <option value="Student">Student</option>
                        <option value="Salaried">Salaried</option>
                        <option value="Self-Employed">Self-Employed</option>
                        <option value="Other">Other</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Income Score (0-100)</label>
                    <input type="number" id="income" name="income" class="input-control" value="50" min="0" max="100" required>
                </div>

                <button type="submit" class="btn-predict">
                    <i class="fa-solid fa-wand-magic-sparkles"></i> Predict Admission
                </button>
            </form>
        </div>

        <!-- Dashboard Analytics Section -->
        <div class="analytics-grid">
            
            <!-- Overall Prediction Banner -->
            <div class="glass-card full-width">
                <div class="result-banner">
                    <span style="color: var(--text-muted); text-transform: uppercase; font-size: 0.85rem; letter-spacing: 1px;">
                        Prediction Status Output
                    </span>
                    <div id="predictionResult" class="status-badge status-yes">
                        AWAITING INPUT
                    </div>
                </div>
            </div>

            <!-- Chart 1: Candidate Input Radar -->
            <div class="glass-card">
                <h4 style="margin-bottom: 1rem; color: var(--text-main);"><i class="fa-solid fa-chart-pie"></i> Feature Profile Radar</h4>
                <div class="chart-container">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>

            <!-- Chart 2: Model Decision Probability -->
            <div class="glass-card">
                <h4 style="margin-bottom: 1rem; color: var(--text-main);"><i class="fa-solid fa-chart-bar"></i> Admission Probability</h4>
                <div class="chart-container">
                    <canvas id="barChart"></canvas>
                </div>
            </div>

            <!-- Chart 3: Feature Weight Analysis -->
            <div class="glass-card full-width">
                <h4 style="margin-bottom: 1rem; color: var(--text-main);"><i class="fa-solid fa-diagram-project"></i> Feature Importance Breakdown</h4>
                <div class="chart-container">
                    <canvas id="importanceChart"></canvas>
                </div>
            </div>

        </div>
    </div>

    <script>
        // Dynamic Theme Switcher
        function setTheme(themeName) {
            document.documentElement.setAttribute('data-theme', themeName);
            document.querySelectorAll('.theme-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.innerText.toLowerCase() === themeName) {
                    btn.classList.add('active');
                }
            });
            updateChartsTheme();
        }

        // Global Chart Instances
        let radarChart, barChart, importanceChart;

        function initCharts() {
            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            const ctxBar = document.getElementById('barChart').getContext('2d');
            const ctxImportance = document.getElementById('importanceChart').getContext('2d');

            radarChart = new Chart(ctxRadar, {
                type: 'radar',
                data: {
                    labels: ['Age', 'Gender', 'Region', 'Occupation', 'Income'],
                    datasets: [{
                        label: 'Candidate Metrics',
                        data: [22, 1, 1, 0, 50],
                        backgroundColor: 'rgba(0, 242, 254, 0.2)',
                        borderColor: '#00f2fe',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { r: { angleLines: { color: 'rgba(255,255,255,0.1)' }, grid: { color: 'rgba(255,255,255,0.1)' } } }
                }
            });

            barChart = new Chart(ctxBar, {
                type: 'doughnut',
                data: {
                    labels: ['Admission Granted (Yes)', 'Admission Denied (No)'],
                    datasets: [{
                        data: [50, 50],
                        backgroundColor: ['#10b981', '#ef4444']
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });

            importanceChart = new Chart(ctxImportance, {
                type: 'bar',
                data: {
                    labels: ['Age', 'Gender', 'Region', 'Occupation', 'Income'],
                    datasets: [{
                        label: 'Decision Tree Weight',
                        data: [0.35, 0.10, 0.15, 0.10, 0.30],
                        backgroundColor: ['#00f2fe', '#4facfe', '#f72585', '#7209b7', '#3b82f6']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { y: { beginAtZero: true } }
                }
            });
        }

        function updateChartsTheme() {
            if(radarChart) radarChart.update();
            if(barChart) barChart.update();
            if(importanceChart) importanceChart.update();
        }

        // AJAX Prediction Form Handling
        document.getElementById('predictionForm').addEventListener('submit', async function (e) {
            e.preventDefault();
            
            const formData = {
                age: parseFloat(document.getElementById('age').value),
                gender: document.getElementById('gender').value,
                region: document.getElementById('region').value,
                occupation: document.getElementById('occupation').value,
                income: parseFloat(document.getElementById('income').value)
            };

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            const resultElem = document.getElementById('predictionResult');
            if (result.prediction === 'yes') {
                resultElem.innerText = "QUALIFIED (YES)";
                resultElem.className = "status-badge status-yes";
            } else {
                resultElem.innerText = "NOT QUALIFIED (NO)";
                resultElem.className = "status-badge status-no";
            }

            // Update Radar Chart Data
            radarChart.data.datasets[0].data = [
                formData.age, 
                result.encoded_features[1] * 30, 
                result.encoded_features[2] * 30, 
                result.encoded_features[3] * 25, 
                formData.income
            ];
            radarChart.update();

            // Update Probability Chart Data
            barChart.data.datasets[0].data = [
                result.probability.yes * 100, 
                result.probability.no * 100
            ];
            barChart.update();
        });

        window.onload = initCharts;
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_LAYOUT)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Preprocess Inputs according to the trained pipeline mappings
    age = float(data.get("age", 0))
    gender = GENDER_MAP.get(data.get("gender"), 0)
    region = REGION_MAP.get(data.get("region"), 0)
    occupation = OCCUPATION_MAP.get(data.get("occupation"), 0)
    income = float(data.get("income", 0))

    features = np.array([[age, gender, region, occupation, income]])

    if model is not None:
        prediction_raw = model.predict(features)[0]
        prediction = str(prediction_raw).lower()

        # Calculate prediction probabilities if supported by the decision tree
        try:
            probs = model.predict_proba(features)[0]
            prob_dict = {"no": float(probs[0]), "yes": float(probs[1])}
        except Exception:
            prob_dict = {"yes": 1.0 if prediction == "yes" else 0.0, "no": 0.0 if prediction == "yes" else 1.0}
    else:
        # Fallback simulation if model.pkl is missing
        prediction = "yes" if income > 40 else "no"
        prob_dict = {"yes": 0.85 if prediction == "yes" else 0.15, "no": 0.15 if prediction == "yes" else 0.85}

    return jsonify({
        "prediction": prediction,
        "probability": prob_dict,
        "encoded_features": [age, gender, region, occupation, income]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
