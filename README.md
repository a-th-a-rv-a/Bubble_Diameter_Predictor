# Bubble Diameter Predictor — Machine Learning for Fluidized Beds

Live Deployed Project Link: https://bubblediameterpredictor-dicrjbcyjet8cezi6azvq4.streamlit.app/

## Project Overview

This project focuses on **predicting bubble size in gas-solid bubbling fluidized beds** — a key factor influencing reactor performance in chemical engineering operations such as combustion and gasification.

Traditional empirical models often fail to capture the nonlinear complexity of bubble formation.  
To overcome these limitations, this project applies **machine learning techniques** to develop accurate, data-driven predictive models.

___________________________________________________________________________________________


## Objectives

- Apply and compare regression models:  
  **Simple Linear Regression (SLR)**, **Multiple Linear Regression (MLR)**, **Support Vector Regression (SVR)**, **Random Forest (RF)**, and **CatBoost Regressor**  
- Evaluate models using **MAE**, **RMSE**, and **R² score**
- Deploy the best-performing model using **Streamlit** for real-time prediction

___________________________________________________________________________________________


## Methodology

### **Data Source**
Dataset obtained from **Harvard Dataverse**:  
[Bubble Size in Fluidized Beds Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VE977W)

___________________________________________________________________________________________


### **Data Preprocessing**
- Removal of missing values and duplicates  
- Outlier detection using *Z-score* and *IQR* methods  
- Normalization and feature scaling  
- Dimensionality reduction via **Principal Component Analysis (PCA)**  

___________________________________________________________________________________________


### **Model Development**
Five models were trained and tested on both the **original** and **PCA-transformed** datasets:

| Model | Type | Description |
|--------|------|-------------|
| Simple Linear Regression | Linear | Baseline benchmark |
| Multiple Linear Regression | Linear | Captures additive effects |
| Support Vector Regression | Non-linear | Kernel-based modeling |
| Random Forest | Ensemble | 30 trees, min leaf size = 8 |
| CatBoost | Gradient Boosting | Handles categorical features |

___________________________________________________________________________________________


## Results and Discussion

| Model | Dataset | R² | RMSE | MAE |
|--------|----------|----|------|-----|
| SLR | Original | 0.397 | 0.7961 | 0.6302 |
| MLR | PCA | 0.914 | 0.3014 | 0.2283 |
| SVR | PCA | 0.951 | 0.2278 | 0.1268 |
| Random Forest | PCA | 0.951 | 0.2277 | 0.1196 |
| **CatBoost (Categorical)** | Combined | **0.981** | **0.0049** | **0.0033** |

___________________________________________________________________________________________


**Key Insights**
- CatBoost achieved the **highest accuracy (R² = 0.981)**.  
- PCA significantly improved model performance and reduced training time.  
- Tree-based models handled nonlinear dependencies more effectively than linear methods.  
- CatBoost excelled due to its ability to handle categorical variables natively.

___________________________________________________________________________________________


## Deployment

A **Streamlit web app** was developed for real-time bubble diameter prediction.

**Live App:** [Bubble Diameter Predictor]([https://bubblediameterpredictor.streamlit.app/](https://bubblediameterpredictor-dicrjbcyjet8cezi6azvq4.streamlit.app/
))  


**App Features**
- Accepts user input for process parameters  
- Predicts bubble diameter instantly using CatBoost model  
- Clean, interactive interface built in **Streamlit**

___________________________________________________________________________________________


## Future Scope

- Expand dataset with broader experimental conditions  
- Explore **deep learning models** (ANN, CNN) for nonlinear pattern detection  
- Compare ML predictions with **empirical correlations**  
- Extend deployment to an **interactive industrial dashboard**


