# 🌿 Leaf Disease Detection Flask App

A deep learning-powered web application to detect plant leaf diseases using image classification. This app helps farmers and agriculturists identify potential plant diseases quickly using a simple, intuitive interface.

## 🧪 About the Project

This project uses a **Convolutional Neural Network (CNN)** built on **MobileNetV2** to classify leaf images into various disease categories. The model is integrated into a **Flask** web app with a clean UI, allowing users to upload leaf images and receive real-time predictions.

---

## 🔍 Features

- Upload plant leaf images directly through the browser
- Classifies among multiple disease types (e.g., Early Blight, Bacterial Spot, Healthy)
- Confidence score included in predictions
- Animated interface with a modern green-themed design
- Responsive web layout with image and prediction display

---

## 💡 Technologies Used

- **Python**   
- **TensorFlow / Keras**   
- **Flask** 
- **HTML5 / CSS3 / JavaScript**  
- **MobileNetV2 (Transfer Learning)**  
- **Heroku / Localhost for deployment**  
- **NumPy / Pillow / H5PY / SciPy**

---

## 🛠️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/sruthidoppasani/leaf-disease-detection.git
cd leaf-disease-detection

```

2. **Create a virtual environment**
```bash
python -m venv leaf-env
source leaf-env/bin/activate    # On Windows: leaf-env\\Scripts\\activate
Install dependencies
```
3. **Install dependencies**
```bash

pip install -r requirements.txt
Prepare your dataset
```
4. **Prepare your dataset**
```bash
Place your Dataset/ folder in the root directory, structured like:
Dataset/
 ├── Tomato_Bacterial_spot/
 ├── Tomato_healthy/
 ├── Potato_Late_blight/
 ...
```
5. **Train the model**
```bash

python train_model.py
```
6. **Run the Flask app**
```bash
python app.py
Then open: http://127.0.0.1:5000
```
