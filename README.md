# 🍷 Wine Quality Classifier

This project is a Machine Learning-based web app that classifies **red wine quality** as either **Good** or **Bad** based on its chemical properties. It uses a `Random Forest Classifier`, is trained on a balanced version of the red wine quality dataset, and is deployed with an interactive **Gradio** interface.

---

## 📌 Problem Statement

Wine quality depends on various physicochemical features. However, manually analyzing these to determine quality is inefficient and inconsistent. This project aims to automate wine quality classification (Good/Bad) based on numerical chemical features using machine learning.

---

## 🚀 Demo

🔗 [Live App on Hugging Face Spaces](#)  

---

## 🔍 Features

- Binary classification: `Good` (quality ≥ 7) vs `Bad` (quality < 7)
- Handles imbalanced data using `RandomOverSampler`
- Trained with `Random Forest Classifier`
- Real-time predictions via Gradio UI
- Pie chart visualization of prediction confidence

---

## 📊 Dataset

- **Name:** Red Wine Quality Dataset  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)  
- **File used:** `wine_quality_classifier.csv`

---

## 🧠 Model & Tools

| Component        | Description                            |
|------------------|----------------------------------------|
| Model            | RandomForestClassifier (Scikit-learn)  |
| Oversampling     | RandomOverSampler from imblearn        |
| User Interface   | Gradio                                 |
| Visualization    | Matplotlib (Pie Chart)                 |
| Programming Lang | Python 3.x                             |

---

## 📈 Accuracy

The model achieved an accuracy of approximately: 96.58 % 

---
  
### LinkedIn Profile

🔗 [LinkedIn Profile](https://www.linkedin.com/in/mohd-altamash-0997592a6?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)

---

### Licence & Author

This project is licensed under the [MIT License](LICENSE).  
You're free to use and modify it, but *you must give credit* to the original author: **Mohd Altamash**.

