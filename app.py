# app.py
from flask import Flask, render_template, request, jsonify
from joblib import load

app = Flask(__name__)

# Static route configuration
app.static_url_path = '/static'
app.static_folder = 'static'

# Load the SVM model
model = load('svm_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the request
        features = [float(request.form.get('sepal_length')),
                    float(request.form.get('sepal_width')),
                    float(request.form.get('petal_length')),
                    float(request.form.get('petal_width'))]

        # Make prediction using the loaded model
        prediction = model.predict([features])[0]

        # Return the prediction result as JSON
        return jsonify({'result': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
