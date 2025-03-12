# Yelp: MLOps Pipeline for Parking & Image Labeling Predictions

## 📌 Project Overview  
This project leverages **Machine Learning (ML) and Deep Learning (DL)** to predict:  
1. **Validated parking availability** for businesses using Yelp dataset features.  
2. **Image labels** from the Yelp Photos dataset using Convolutional Neural Networks (CNN).  

We implement **MLOps best practices** for model building, tuning, interpretation, and deployment using **Flask, Gunicorn, Streamlit, and Hugging Face Spaces**.  

---

## 📊 **Objectives**  
✅ **Perform Data Analysis & Visualization** 
✅ **Preprocess Data & Engineer Features**  
✅ **Train & Tune ML Models** (XGBoost, AdaBoost)
✅ **Train & Tune DNN Model** for parking prediction  
✅ **Interpretability for Model Tuning** 
✅ **Train DNN & CNN models** for image labeling  
✅ **Deploy models using Flask/Gunicorn, Streamlit, or Hugging Face**  

---

## 🏗 **Project Workflow**  
1. **Data Collection & Preprocessing**  
   - Download Yelp dataset: [Yelp Open Dataset](https://www.yelp.com/dataset/download)  
   - Clean and preprocess data for feature extraction.  

2. **Data Analysis & Feature Engineering**  
   - Identify key features affecting parking validation.  
   - Perform **exploratory data analysis (EDA)** and visualizations.  

3. **Model Development & Evaluation**  
   - Train and tune **XGBoost & AdaBoost** models for parking prediction.  
   - Train **DNN Model** for parking prediction (evaluate with Confusion Matrix & AUC).  
   - Train **CNN Model** for image classification (evaluate with Confusion Matrix & AUC).  

4. **Model Interpretation & Tuning**  
   - Use **SHAP, LIME, or feature importance** to understand model decisions.  
   - Improve model performance based on insights.  

5. **Deployment & MLOps**  
   - Deploy models via **Flask/Gunicorn & Streamlit/Hugging Face**.  
   - Ensure model outputs are **interpretable & user-friendly**.  

---

## 🔧 **Tech Stack**  
**📊 Data Processing & Visualization:** Pandas, NumPy, Matplotlib, Seaborn  
**⚡ ML & Deep Learning:** XGBoost, AdaBoost, TensorFlow, Keras  
**🖼 Computer Vision:** CNN (Convolutional Neural Networks)  
**🛠 Deployment:** Flask, Gunicorn, Streamlit, Hugging Face  
**🚀 MLOps Practices:** Model interpretability, version control, CI/CD  

---
## 📂 Project Structure
<pre>
MLOps-Pipeline-Parking-Image-Labeling-Predictions/ 
│── CNN_Model_Evaluation_Interpretability.html 
│── DNN_Model_Evaluation_Interpretability.html 
│── DNN_predicting_validated_parking.h5 
│── DNN_predicting_validated_parking.keras 
│── XGBoost_Build_Tune_Model.html 
│── xgboost_model.pkl
│── README.md 
│── requirements.txt 
│── app.py 
│── .gitattributes 
</pre>

## 🚀 **How to Run the Project**  

### 1️⃣ **Install Dependencies**  
```
pip install -r requirements.txt
```

### 2️⃣ Run the Model API  

#### **Option 1: If using Gradio**  
Run:

```
python app.py
```
#### **Option 2: If using Flask with Gunicorn**  
Run:

```
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### **Option 3: If using Gradio**  
Run:

```
python app.py
```
This will launch a **Gradio web interface** in your browser.

📌 **Key Insights from the Analysis**
- 🔹 Businesses with higher ratings & more reviews are more likely to offer validated parking.
- 🔹 Features like price level, credit card acceptance, and check-ins significantly impact parking availability.
- 🔹 CNN models can effectively classify business photos based on Yelp image data.

🎯 **Future Improvements**
- Implement real-time model monitoring for deployed models.
- Fine-tune hyperparameters for improved model accuracy.
- Deploy a REST API for external integrations.

💡 **Contributions**
- Feel free to fork, contribute, or raise an issue! 🚀

📩 **Let's connect on LinkedIn**: https://www.linkedin.com/in/sweatha-palani/

#AI #MachineLearning #MLOps #DeepLearning #XGBoost #ComputerVision #Flask #Streamlit #YelpDataset 🚀
