from flask import Flask, request, render_template, redirect, url_for, send_file
import sqlite3
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
import joblib  # Assuming joblib is used to load the pre-trained model and encoders


app = Flask(__name__)

# Load the trained model and scaler
# Load the saved models and preprocessors
ensemble_model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Connect to SQLite database
def get_db_connection():
    conn = sqlite3.connect('heart_disease.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')
remedies = {
    'high_chol': "Your cholesterol level is high. Consider reducing saturated fats, eliminating trans fats, eating omega-3 fatty acids, increasing soluble fiber, and adding whey protein to your diet.",
    'high_bps': "Your blood pressure is high. Consider eating a healthier diet with less salt, exercising regularly, maintaining a healthy weight, and managing stress.",
    'low_thalach': "Your maximum heart rate is low. Regular cardiovascular exercise can help improve your heart rate.",
    'high_oldpeak': "Your ST depression is high. This can indicate possible ischemia. Consult your doctor for a more detailed diagnosis and potential lifestyle changes."
}




# Pre-load the necessary components
ensemble_model = joblib.load('ensemble_model.pkl')  
scaler = joblib.load('scaler.pkl')  # Load the pre-fitted scaler
label_encoders = joblib.load('label_encoders.pkl')  # Load pre-fitted label encoders
X_train = joblib.load('X_train.pkl')
y_train = joblib.load('y_train.pkl')


# Assuming X_train and y_train are available for model calibration (normally, you'd have this done during training)
# X_train and y_train should be your training data. If not available, ensure calibration is pre-done and saved.
# For the sake of this example, let's assume these are pre-calibrated and stored for later use
calibrated_ensemble_model = CalibratedClassifierCV(ensemble_model, method='isotonic', cv=5)

calibrated_ensemble_model.fit(X_train, y_train)  # This step would ideally be done during the training phase
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            age = float(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = float(request.form['trestbps'])
            chol = float(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = float(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            
            # Create DataFrame for model input
            input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]],
                                      columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope'])
            
            # Convert input data to appropriate data types
            input_data = input_data.astype({
                'age': 'float64', 
                'sex': 'category', 
                'cp': 'category', 
                'trestbps': 'float64', 
                'chol': 'float64', 
                'fbs': 'category', 
                'restecg': 'category', 
                'thalach': 'float64', 
                'exang': 'category', 
                'oldpeak': 'float64', 
                'slope': 'category'
            })
            
            # Encode categorical features using pre-loaded encoders
            for feature in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']:
                le = label_encoders[feature]
                input_data[feature] = le.transform(input_data[feature])
            
            # Scale numerical features using pre-loaded scaler
            numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            input_data[numerical_features] = scaler.transform(input_data[numerical_features])
            
            # Predict using the pre-calibrated ensemble model
            prediction_proba = calibrated_ensemble_model.predict_proba(input_data)
            
            # Determine likelihood category based on probability
            proba_percentage = prediction_proba[0][1] * 100  # Assuming positive class is at index 1
            if proba_percentage < 25:
                likelihood = 'Very unlikely (Less than 25% probability)'
            elif proba_percentage < 50:
                likelihood = 'Unlikely (Between 25% and 50% probability)'
            elif proba_percentage < 75:
                likelihood = 'Likely (Between 50% and 75% probability)'
            else:
                likelihood = 'Highly likely (Above 75% probability)'
            
            # Determine remedies based on input values
            user_remedies = []
            if chol > 200:  # Assuming 200 mg/dL as a threshold for high cholesterol
                user_remedies.append(remedies['high_chol'])
            if trestbps > 120:  # Assuming 120 mm Hg as a threshold for high blood pressure
                user_remedies.append(remedies['high_bps'])
            if thalach < 60:  # Assuming 60 bpm as a threshold for low heart rate
                user_remedies.append(remedies['low_thalach'])
            if oldpeak > 2:  # Assuming 2 mm as a threshold for high ST depression
                user_remedies.append(remedies['high_oldpeak'])

            return render_template('result.html', prediction=likelihood, remedies=user_remedies)
        
        except Exception as e:
            # Handle any errors that occur during the prediction process
            error_message = f"An error occurred during the prediction: {str(e)}"
           # return render_template('error.html', error_message=error_message)
    
    return render_template('predict.html')


@app.route('/update', methods=['GET', 'POST'])
def update():
    if request.method == 'POST':
        # Get form data
        age = request.form['age']
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        target = request.form['target']

        # Connect to database and insert data
        conn = get_db_connection()
        conn.execute('''INSERT INTO patients (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, target) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, target))
        conn.commit()
        conn.close()

        return redirect(url_for('index'))
    
    return render_template('update.html')

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    
    if request.method == 'POST':
        # Get form data
        age = request.form['age']
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']

        # Create DataFrame for comparison
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]],
                                  columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope'])

        # Convert columns to numeric
        input_data = input_data.apply(pd.to_numeric)

        # Retrieve existing patient data with and without disease
        conn = get_db_connection()
        disease_data = pd.read_sql_query('SELECT * FROM patients WHERE target=1', conn)
        no_disease_data = pd.read_sql_query('SELECT * FROM patients WHERE target=0', conn)
        conn.close()

        # Create visualizations
        fig, axes = plt.subplots(4, 3, figsize=(15, 15))
        axes = axes.flatten()

        for i, column in enumerate(input_data.columns):
            sns.histplot(disease_data[column], kde=True, ax=axes[i], color='blue', label='Disease')
            sns.histplot(no_disease_data[column], kde=True, ax=axes[i], color='green', label='No Disease')
            axes[i].axvline(input_data[column][0], color='red', linestyle='dashed', linewidth=2, label='Input Patient')
            axes[i].set_title(column)
            axes[i].legend()

        plt.tight_layout(pad=4.0)  # Increase padding

        # Save the histogram plot to a BytesIO object
        hist_img = io.BytesIO()
        plt.savefig(hist_img, format='png')
        hist_img.seek(0)
        hist_plot_url = base64.b64encode(hist_img.getvalue()).decode()

        # Create scatter plot for selected features
        scatter_features = ['age', 'chol', 'thalach', 'oldpeak']
        scatter_fig, scatter_axes = plt.subplots(2, 2, figsize=(10, 10))
        scatter_axes = scatter_axes.flatten()

        for i, feature in enumerate(scatter_features):
            scatter_axes[i].scatter(disease_data[feature], disease_data['target'], color='blue', label='Disease')
            scatter_axes[i].scatter(no_disease_data[feature], no_disease_data['target'], color='green', label='No Disease')
            scatter_axes[i].scatter(input_data[feature], 1, color='red', marker='x', s=100, label='Input Patient')
            scatter_axes[i].set_title(f'{feature} vs Target')
            scatter_axes[i].set_xlabel(feature)
            scatter_axes[i].set_ylabel('Target')
            scatter_axes[i].legend()

        plt.tight_layout(pad=4.0)  # Increase padding

        # Save the scatter plot to a BytesIO object
        scatter_img = io.BytesIO()
        scatter_fig.savefig(scatter_img, format='png')
        scatter_img.seek(0)
        scatter_plot_url = base64.b64encode(scatter_img.getvalue()).decode()

        return render_template('compare_result.html', hist_plot_url=hist_plot_url, scatter_plot_url=scatter_plot_url)
    
    return render_template('compare.html')
from flask import Flask, request, render_template, redirect, url_for, send_file, send_from_directory

# Add route to serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
