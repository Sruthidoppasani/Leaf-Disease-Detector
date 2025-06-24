from flask import Flask, render_template, request, redirect, url_for
import os
import uuid
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model('leaf_model.h5')

# Load your saved class labels
import numpy as np
class_labels = np.load('class_labels.npy', allow_pickle=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return "❌ No file uploaded"

    # Save uploaded image with unique name
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_class = class_labels[class_index]
    confidence = float(np.max(prediction) * 100)

    # Pass results to result.html
    return render_template(
        'result.html',
        prediction=f"{predicted_class} ({confidence:.2f}%)",
        image_url=url_for('static', filename=f'uploads/{filename}')
    )

if __name__ == '__main__':
    app.run(debug=True)
