from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'churn_model.pkl')
model = joblib.load(MODEL_PATH)

# Feature order must match training data
FEATURES = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'Contract_Month-to-month',
    'Contract_One year',
    'Contract_Two year',
    'InternetService_Fiber optic',
    'InternetService_DSL',
    'PaymentMethod_Electronic check',
    'SeniorCitizen',
    'Dependents',
    'PaperlessBilling',
    'TechSupport',
    'OnlineBackup',
]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect and sanitize form inputs
        tenure = float(request.form['tenure'])
        monthly_charges = float(request.form['monthly_charges'])
        total_charges = float(request.form['total_charges'])
        contract = request.form['contract']
        internet = request.form['internet_service']
        payment = request.form['payment_method']
        senior = int(request.form.get('senior_citizen', 0))
        dependents = int(request.form.get('dependents', 0))
        paperless = int(request.form.get('paperless_billing', 0))
        tech_support = int(request.form.get('tech_support', 0))
        online_backup = int(request.form.get('online_backup', 0))

        # Encode categoricals
        features = [
            tenure,
            monthly_charges,
            total_charges,
            1 if contract == 'Month-to-month' else 0,
            1 if contract == 'One year' else 0,
            1 if contract == 'Two year' else 0,
            1 if internet == 'Fiber optic' else 0,
            1 if internet == 'DSL' else 0,
            1 if payment == 'Electronic check' else 0,
            senior,
            dependents,
            paperless,
            tech_support,
            online_backup,
        ]

        X = np.array(features).reshape(1, -1)
        probability = model.predict_proba(X)[0][1]
        prediction = 'HIGH RISK' if probability >= 0.5 else 'LOW RISK'
        pct = round(probability * 100, 1)

        return render_template(
            'result.html',
            prediction=prediction,
            probability=pct,
            tenure=tenure,
            monthly_charges=monthly_charges,
        )

    except Exception as e:
        return render_template('result.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=False)
