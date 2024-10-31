from flask import Flask, render_template, request, redirect, url_for, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the trained model
# The pretrained model is used, but the other model can also be used
model = load_model('models/best_model_pretrained.keras')

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(640, 640, 3))  
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and classification
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image and predict
        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)
        
        # three classes: ['Capacitor', 'IC', 'Resistor']
        classes = ['Capacitor', 'IC', 'Resistor']
        predicted_class = classes[np.argmax(prediction)]
        
        # Return the result as JSON
        return jsonify({
            'class': predicted_class,
            'confidence': float(np.max(prediction))
        })
    else:
        return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)
