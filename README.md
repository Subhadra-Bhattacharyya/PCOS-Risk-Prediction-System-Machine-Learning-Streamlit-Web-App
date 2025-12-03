# 🩺 PCOS Risk Prediction System  
A Machine-Learning–based Web Application for PCOS Risk Assessment  
Built with **Python**, **Streamlit**, **scikit-learn**, and **SHAP**.

---

## 📌 Overview  
Polycystic Ovary Syndrome (PCOS) is a common endocrine disorder that affects women of reproductive age. Traditional diagnosis requires multiple clinical and biochemical tests, making early detection challenging.

This project provides an interactive **PCOS Risk Prediction System** that estimates the probability of PCOS using demographic data, symptoms, lifestyle indicators, hormonal values, and ultrasound findings. The model is deployed through a Streamlit interface and includes SHAP-based explainability to interpret predictions.

---

## 🚀 Features  
### ✔ **Interactive Streamlit UI**  
User-friendly interface to input medical and lifestyle data.

### ✔ **Real-Time PCOS Prediction**  
Predicts:  
- **PCOS / No PCOS**  
- **Risk probability (0–1)**  
- **Risk classification**  
  - Low risk  
  - Moderate risk  
  - High risk  

### ✔ **Automated Calculations**  
- BMI (Body Mass Index)  
- WHR (Waist–Hip Ratio)

### ✔ **Explainable AI (SHAP)**  
Displays top contributing features that influenced the prediction.

### ✔ **Handles Missing Data**  
Missing values are automatically imputed using the model pipeline.

### ✔ **Anonymous Logging (Optional)**  
User can allow saving anonymized inputs for analytics or model improvement.

---

## 🧠 Machine Learning Pipeline  
The model and preprocessing steps are packaged in:

