from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import shap
from io import BytesIO
import base64
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

app = Flask(__name__)
app.secret_key = 'WakeUpCall_Prototype'

# Load all model files 
model = joblib.load("C:/Users/User/Desktop/COLLEGE/WakeUpCallPrototype/models/sleep_disorder_multiclass_lgbm_model (1).pkl")
scaler = joblib.load("C:/Users/User/Desktop/COLLEGE/WakeUpCallPrototype/models/sleep_disorder_scaler.pkl")
label_encoder = joblib.load("C:/Users/User/Desktop/COLLEGE/WakeUpCallPrototype/models/sleep_disorder_label_encoder.pkl")
feature_names = joblib.load("C:/Users/User/Desktop/COLLEGE/WakeUpCallPrototype/models/sleep_disorder_feature_names.pkl")
label_mappings = joblib.load("C:/Users/User/Desktop/COLLEGE/WakeUpCallPrototype/models/categorical_label_mappings.pkl")
explainer = joblib.load("C:/Users/User/Desktop/COLLEGE/WakeUpCallPrototype/models/sleep_disorder_shap_explainer.pkl")

# Helper function for BMI classification (Filipino standards)
# def classify_bmi(bmi):
#     if bmi < 18.5: return 'Underweight'
#     elif 18.5 <= bmi < 23: return 'Normal'
#     elif 23 <= bmi < 27.5: return 'Overweight'
#     else: return 'Obese'

# def get_top_features(shap_values, class_idx, sample_values, top_n=5):
#     class_shap = shap_values[class_idx][0]
#     features = list(zip(feature_names, class_shap, sample_values))
#     features.sort(key=lambda x: abs(x[1]), reverse=True)
    
#     top_features = []
#     for name, shap_val, value in features[:top_n]:
#         if name == 'BMI Category':
#             bmi_mapping = {v: k for k, v in label_mappings['BMI Category'].items()}
#             formatted_name = f"BMI ({bmi_mapping.get(int(value), 'Unknown')})"
#         elif name == 'Sleep Duration':
#             formatted_name = f"Sleep Duration ({value:.1f} hrs)"
#         elif name in ['BP_Systolic', 'BP_Diastolic']:
#             systolic_idx = feature_names.index('BP_Systolic')
#             diastolic_idx = feature_names.index('BP_Diastolic')
#             formatted_name = f"BP ({sample_values[systolic_idx]:.0f}/{sample_values[diastolic_idx]:.0f})"
#         elif name == 'Stress Level':
#             formatted_name = f"Stress Level ({int(value)}/10)"
#         else:
#             formatted_name = name.replace('_', ' ').title()
        
#         impact = shap_val * 100
#         top_features.append({
#             'name': formatted_name,
#             'impact': abs(round(impact, 1)),
#             'direction': 'increases risk' if impact > 0 else 'decreases risk'
#         })
#     return top_features

# Routes
# ----- HEADERS -----
@app.route('/', methods=['GET'])
def home():
    session.clear()
    return render_template('index.html')
@app.route('/aboutOSA', methods=['GET'])
def aboutOSA():
    return render_template('aboutOSA.html')
@app.route('/aboutus', methods=['GET'])
def aboutus():
    return render_template('aboutus.html')

# ----- FOOOTER -----
@app.route('/privacypolicy', methods=['GET'])
def privacypolicy():
    return render_template('privacypolicy.html')
@app.route('/contactus', methods=['GET'])
def contactus():
    return render_template('contactus.html')
@app.route('/faq', methods=['GET'])
def faq():
    return render_template('faq.html')

# ----- BODY -----
@app.route('/consent', methods=['GET'])
def consent():
    return render_template('consent.html')

# ----- FORMS -----
@app.route('/demographic', methods=['GET', 'POST'])
def demographic():
    # if request.method == 'POST':
    #     try:
    #         session['demographic_data'] = {
    #             'Gender': request.form['Gender'],
    #             'Age': float(request.form['age']),
    #             'Occupation': request.form['occupation']
    #         }
    #         return redirect(url_for('sleep'))
    #     except Exception as e:
    #         return render_template('demographic.html', error=str(e))
    return render_template('demographic.html')

@app.route('/sleep', methods=['GET', 'POST'])
def sleep():
    # if request.method == 'POST':
    #     try:
    #         session['sleep_data'] = {
    #             'Sleep Duration': float(request.form['Sleep Duration']),
    #             'Quality of Sleep': float(request.form['Quality of Sleep']),
    #             'Snoring': request.form.get('snoring', 'no')
    #         }
    #         return redirect(url_for('lifestyle'))
    #     except Exception as e:
    #         return render_template('sleep.html', error=str(e))
    return render_template('sleep.html')

@app.route('/lifestyle', methods=['GET', 'POST'])
def lifestyle():
    # if request.method == 'POST':
    #     try:
    #         session['lifestyle_data'] = {
    #             'Physical Activity Level': float(request.form['Physical Activity Level']),
    #             'Stress Level': float(request.form['Stress Level']),
    #             'Daily Steps': float(request.form['Daily Steps'])
    #         }
    #         return redirect(url_for('health'))
    #     except Exception as e:
    #         return render_template('lifestyle.html', error=str(e))
    return render_template('lifestyle.html')

@app.route('/health', methods=['GET', 'POST'])
def health():
    # if request.method == 'POST':
    #     try:
    #         # Process blood pressure
    #         bp = request.form['Blood Pressure'].split('/')
    #         systolic, diastolic = float(bp[0]), float(bp[1]) if len(bp) > 1 else 0
            
    #         # Calculate BMI
    #         height = float(request.form['Height']) / 100  # cm to m
    #         weight = float(request.form['Weight'])
    #         bmi = weight / (height ** 2)
    #         bmi_category = classify_bmi(bmi)
            
    #         # Store health data
    #         session['health_data'] = {
    #             'Height': height * 100,  # Store in cm
    #             'Weight': weight,
    #             'BMI Category': bmi_category,
    #             'BP_Systolic': systolic,
    #             'BP_Diastolic': diastolic,
    #             'Heart Rate': float(request.form['Heart Rate'])
    #         }
            
    #         # Prepare feature vector in EXACT model training order
    #         all_data = {
    #             **session.get('demographic_data', {}),
    #             **session.get('sleep_data', {}),
    #             **session.get('lifestyle_data', {}),
    #             **session.get('health_data', {})
    #         }
            
    #         feature_order = feature_names  # Use the same order as training
    #         new_features = []
    #         for feature in feature_order:
    #             if feature in ['Gender', 'Occupation', 'BMI Category']:
    #                 mapping = label_mappings[feature]
    #                 value = str(all_data.get(feature, '')).title()
    #                 new_features.append(mapping.get(value, 0))
    #             else:
    #                 new_features.append(float(all_data.get(feature, 0)))
            
    #         # Scale and predict
    #         scaled_features = scaler.transform([new_features])
    #         probabilities = model.predict(scaled_features)[0]
    #         predicted_class = np.argmax(probabilities)
    #         predicted_label = label_encoder.inverse_transform([predicted_class])[0]
            
    #         # SHAP analysis
    #         explainer = shap.TreeExplainer(model)
    #         shap_values = explainer.shap_values(scaled_features)
            
    #         # Create visualization
    #         plt.figure()
    #         shap.force_plot(
    #             explainer.expected_value[predicted_class],
    #             shap_values[predicted_class][0],
    #             scaled_features[0],
    #             feature_names=feature_names,
    #             matplotlib=True,
    #             show=False
    #         )
    #         buf = BytesIO()
    #         plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    #         plt.close()
    #         shap_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
            
    #         # Prepare results
    #         results = {
    #             'prediction': predicted_label,
    #             'confidence': round(probabilities[predicted_class] * 100, 2),
    #             'probabilities': {
    #                 label: round(prob * 100, 2) 
    #                 for label, prob in zip(label_encoder.classes_, probabilities)
    #             },
    #             'shap_plot': shap_plot,
    #             'top_features': get_top_features(shap_values, predicted_class, scaled_features[0]),
    #             'bmi': round(bmi, 1),
    #             'bmi_category': bmi_category
    #         }
            
    #         session['results'] = results
    #         return redirect(url_for('result'))
            
    #     except Exception as e:
    #         print(f"Error: {str(e)}")
    #         return render_template('health.html', error=str(e))
    
    return render_template('health.html')

@app.route('/result')
def result():
    return render_template('result.html')
    # results = session.get('results')
    # if not results:
    #     return redirect(url_for('home'))
    
    # # Determine risk level
    # confidence = results['confidence']
    # if confidence >= 70: risk_level = "High"
    # elif confidence >= 40: risk_level = "Moderate"
    # else: risk_level = "Low"
    
    # return render_template('result.html', 
    #                      results=results,
    #                      risk_level=risk_level)



if __name__ == '__main__':
    app.run(port=3000, debug=True)