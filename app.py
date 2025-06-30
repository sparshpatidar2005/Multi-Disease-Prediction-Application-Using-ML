from flask import Flask, render_template, request
import os
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image

from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Common Disease Model ===
with open("model/mapping.json", "r") as f:
    name_to_id = json.load(f)
    id_to_name = {v: k for k, v in name_to_id.items()}

df = pd.read_csv("model/symptom-disease-train-dataset.csv")
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0]
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

disease_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])
disease_model.fit(df["text"], df["label"])

# === Other Models ===
heart_model = joblib.load("model/heart_disease_xgb_model.joblib")
diabetes_model = CatBoostClassifier()
diabetes_model.load_model("model/catboost_model.cbm")
parkinsons_model = joblib.load("model/knn_parkinsons_model.pkl")
liver_imputer = joblib.load("model/imputer.joblib")
liver_scaler = joblib.load("model/scaler.joblib")
liver_model = RandomForestClassifier()
liver_model.fit([[0] * 10], [0])  # Dummy fit
lung_model = load_model("model/CNN_best_model.keras")

# === Breast Cancer Model (from CSV) ===
df_breast = pd.read_csv("model/breast_cancer_features.csv")
X_breast = df_breast.drop(columns=["label"])
y_breast = df_breast["label"]
breast_model = RandomForestClassifier()
breast_model.fit(X_breast, y_breast)

def preprocess_lung_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_common():
    try:
        user_input = request.form["symptoms"]
        prediction_id = disease_model.predict([user_input])[0]
        prediction_name = id_to_name.get(prediction_id, "Unknown Disease")
        return render_template("result.html", symptoms=user_input, prediction=prediction_name)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/predict-heart-disease", methods=["POST"])
def predict_heart_disease():
    try:
        features = [float(request.form[field]) for field in [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]]
        prediction = heart_model.predict([features])[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        return render_template("result.html", symptoms="Heart Test Values", prediction=result)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/predict-diabetes", methods=["POST"])
def predict_diabetes():
    try:
        features = [float(request.form[field]) for field in [
            "pregnancies", "glucose", "bloodpressure", "skinthickness",
            "insulin", "bmi", "dpf", "age"
        ]]
        prediction = diabetes_model.predict([features])[0]
        result = "Diabetes Detected" if prediction == 1 else "No Diabetes"
        return render_template("result.html", symptoms="Diabetes Test Values", prediction=result)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/predict-parkinsons", methods=["POST"])
def predict_parkinsons():
    try:
        features = [float(request.form[f"f{i}"]) for i in range(22)]
        prediction = parkinsons_model.predict([features])[0]
        result = "Parkinson's Detected" if prediction == 1 else "No Parkinson's"
        return render_template("result.html", symptoms="Parkinson's Features", prediction=result)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/predict-liver", methods=["POST"])
def predict_liver():
    try:
        features = [float(request.form[field]) for field in [
            "age", "gender", "total_bilirubin", "direct_bilirubin",
            "alk_phosphate", "alamine_aminotransferase", "aspartate_aminotransferase",
            "total_proteins", "albumin", "ag_ratio"
        ]]
        imputed = liver_imputer.transform([features])
        scaled = liver_scaler.transform(imputed)
        prediction = liver_model.predict(scaled)[0]
        result = "Liver Disease Detected" if prediction == 1 else "No Liver Disease"
        return render_template("result.html", symptoms="Liver Test Values", prediction=result)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/predict-lung-disease", methods=["POST"])
def predict_lung_disease():
    try:
        file = request.files["image"]
        if file.filename == "":
            return "No file uploaded", 400

        image = Image.open(file.stream)
        processed_image = preprocess_lung_image(image)
        prediction = lung_model.predict(processed_image)
        label = "Lung Disease Detected" if prediction[0][0] > 0.5 else "No Lung Disease"
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        image.save(file_path)
        return render_template("result.html", symptoms="Lung X-ray", prediction=label, image_path=file_path)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/predict-breast-cancer", methods=["POST"])
def predict_breast_cancer():
    try:
        features = [float(request.form[f"feat_{i}"]) for i in range(64)]
        prediction = breast_model.predict([features])[0]
        label = "Malignant" if prediction == 1 else "Benign"
        return render_template("result.html", symptoms="Breast Cancer Features", prediction=label)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
