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
pcos_final_pipeline.joblib
feature_columns.json / feature_columns.joblib


The pipeline includes:
- Missing value imputation  
- Scaling and encoding  
- ML classifier (e.g., Random Forest / XGBoost / Logistic Regression)  
- Feature alignment  

The trained model outputs:
- Binary prediction (PCOS / No PCOS)  
- Probability score  
- Risk category  

---

## 🛠️ Technologies Used  
- **Python 3.x**  
- **Streamlit**  
- **scikit-learn**  
- **NumPy & Pandas**  
- **SHAP (Explainability)**  
- **Joblib** for model loading  



## ▶️ How to Run the Project Locally

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Subhadra-Bhattacharyya/pcos-prediction-system.git
cd pcos-prediction-system

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit App
streamlit run app.py

4️⃣ Open in Browser

Streamlit will automatically open at:

http://localhost:8501

🧬 Input Features Used

The model supports the following categories of data:

• Demographic & Anthropometric

Age, Height, Weight, BMI, Waist, Hip, WHR

• Menstrual & Reproductive

Cycle type (Regular/Irregular), Cycle length, Pregnancy, Abortions, Marriage years

• Symptoms & Lifestyle

Hair loss, Hair growth, Pimples, Skin darkening, Fast food intake, Exercise habits

• Biochemical Parameters (Optional)

FSH, LH, FSH/LH, AMH, TSH, PRL, Hb, Progesterone, RBS

• Ultrasound Findings

Follicle count (L/R), Avg follicle size (L/R), Endometrium thickness

📊 Explainability with SHAP

The app attempts to compute SHAP values for each prediction:

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_trans)


The interface displays top influential features for transparency and clinical insight.

🌐 Demo (Optional)

If you deploy on Streamlit Cloud, Render, or HuggingFace Spaces, you can add your live link here.

🤝 Contributing

Pull requests, issues, and feature suggestions are welcome!
Feel free to fork and enhance the system.

🛡️ Disclaimer

This tool is intended for educational and research purposes only.
It is not a medical diagnostic device.
Users must consult certified healthcare professionals for clinical decisions.

