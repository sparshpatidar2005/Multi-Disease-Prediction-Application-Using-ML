# 🩺 Disease Prediction Web App

An AI-powered web application for predicting a wide range of diseases based on symptoms, as well as specialized modules for heart disease, diabetes, Parkinson's, and liver disease prediction. The project leverages machine learning and deep learning models for fast, reliable, and accessible health assessment.

![App Dashboard](static/illustration.png)

---

## 🚀 Features

- **Common Disease Prediction:** Enter symptoms in plain text and get likely diseases from 1000+ possibilities.
- **Specialized Models:** 
  - Heart Disease (XGBoost)
  - Diabetes (CatBoost)
  - Parkinson’s Disease (KNN)
  - Liver Disease (CatBoost with Imputation & Scaling)
- **Attractive, Modern UI:** Responsive and visually appealing dashboard.
- **Easy Input:** Intuitive forms and defaults for quick predictions.
- **Result Visualization:** Clean display of prediction results.
- **Extensible:** Easily add new models or datasets.

---

## 📚 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Credits & References](#-credits--references)
- [License](#-license)

---

## 🎬 Demo

**[Video Demo](https://youtu.be/rHJJRbHYMRQ?si=gbmUsCid3JMFUq1c)**  
*Watch the demo on YouTube!*

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS (Custom), JavaScript
- **Machine Learning:** scikit-learn, XGBoost, CatBoost, joblib, pickle
- **Data:** Custom symptom-disease datasets, UCI & public datasets for specialized models
- **Deployment:** Local (can be deployed to Heroku, AWS, etc.)

---

## 🏁 Getting Started

### 1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/disease-prediction.git
cd disease-prediction
```

### 2. **Install Requirements**
It's recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```
*(Add or update `requirements.txt` with all needed packages.)*

### 3. **Download/Place Model Files**
Place these files in the `model/` directory:
- `mapping.json`
- `symptom-disease-train-dataset.csv`
- `symptom-disease-test-dataset.csv`
- `heart_disease_xgb_model.joblib`
- `catboost_model.cbm`
- `knn_parkinsons_model.pkl`
- `knn_scaler.pkl`
- `imputer.joblib`
- `scaler.joblib`
- `best_model_checkpoint.pth` *(if needed for other models)*

### 4. **Run the Application**
```bash
python app.py
```
Open [http://localhost:5000] in your browser.

---

## 🧑‍💻 Usage

1. **Home Dashboard:**  
   Choose which disease prediction you want: Common, Heart, Diabetes, Liver, or Parkinson's.

2. **Input Data:**  
   Fill in the form with symptoms or medical data (with many fields auto-filled by default).

3. **Get Results:**  
   The app instantly shows the prediction and the inputs used.

4. **Try Again:**  
   Use the dashboard to switch between diseases or make new predictions.

---

## 🗂️ Project Structure

```
disease-prediction/
│
├── app.py                 # Main Flask app
├── templates/             # HTML files
├── static/                # CSS, JS, images
├── model/                 # Trained model files and datasets
│    ├── heart_disease_xgb_model.joblib
│    ├── catboost_model.cbm
│    ├── knn_parkinsons_model.pkl
│    ├── knn_scaler.pkl
│    ├── imputer.joblib
│    ├── scaler.joblib
│    └── ...
├── requirements.txt       # Python requirements
└── README.md              # Project documentation
```

---

## 🧑‍🎓 Credits & References

- Inspired by open-source projects and datasets from [UCI Machine Learning Repository](https://archive.ics.uci.edu/), Kaggle, and [many others](#).
- See [mapping.json](model/mapping.json) for all supported diseases.

---

## 📄 License

MIT License.  
See [LICENSE](LICENSE) for details.

---

## 🙋‍♂️ Questions? Feedback?

Feel free to open an issue or contact the author via GitHub or LinkedIn.

---

**Happy Predicting! 🚑**
