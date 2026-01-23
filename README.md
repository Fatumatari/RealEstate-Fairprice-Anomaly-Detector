# 🏠 FairPrice: Real Estate Price Fairness Classification

---

## 📌 Project Overview

The real estate market suffers from **pricing opacity**. Buyers, renters, sellers, and agents often rely on intuition or incomplete comparables to judge whether a property is reasonably priced. This can lead to:

* Overpriced listings staying too long on the market  
* Underpriced listings causing seller losses  
* Poor negotiation outcomes  
* Reduced trust in listing platforms  

To address this, we built a **machine learning classification system** that labels property listings as:

* **Underpriced**  
* **Fairly Priced**  
* **Overpriced**  

Unlike traditional price prediction (a noisy regression task), this system focuses on **relative price fairness**, producing outputs that are **interpretable, actionable, and aligned with real-world decision making**.

---

## 🎯 Business Objectives

The project aims to improve pricing transparency in the Kenyan real estate market by:

* Classifying property listings into underpriced, fair, or overpriced categories  
* Supporting better pricing decisions for buyers, renters, sellers, and agents  
* Enabling scalable pricing analysis across cities and property types  
* Providing interpretable outputs suitable for non-technical users (platform operators, real estate professionals)  

---

## 🧠 Why Classification Instead of Regression?

Traditional price prediction models are:

* Highly sensitive to outliers  
* Unstable in heterogeneous markets  
* Hard to interpret for end users  

By reframing pricing as a **multi-class classification problem**, we:

* Increase robustness to noise  
* Align with real-world decision logic  
* Produce directly actionable labels  
* Improve interpretability  

---

## 📂 Dataset Description

The cleaned and feature-engineered dataset contains **9,091 property listings** with fields such as:

* `price` (capped for modeling stability) and `price_raw` (original)  
* Bedrooms, bathrooms, toilets, parking  
* Furnished, serviced, shared  
* Category (For Rent / For Sale), type, sub_type  
* State, locality, listdate  

**Capping Rules Applied:**

| Feature | Lower Bound | Upper Bound |
|---------|------------|------------|
| Bedrooms | 0 | 10 |
| Bathrooms | 0 | 10 |
| Parking | 0 | 10 |
| Rent Price | 5,500 | 600,000 |
| Sale Price | 600,000 | 100,000,000 |

Raw features (*_raw) were retained for **anomaly detection** and interpretability.

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was conducted to understand market structure, pricing dynamics, and feature relationships.

**Key Analyses:**

* Distribution of prices (rent vs sale)  
* Price trends over time  
* Property types and sub-types  
* Geographic price differences (state & locality)  
* Feature distributions (bedrooms, bathrooms, amenities)  
* Correlation heatmaps  
* Category-wise comparisons  

**Key Observations:**

* Heavy price skewness  
* Strong location effects  
* Nonlinear relationships  
* Distinct rental vs sale market regimes  

**Visualizations included:**

* Price distribution histograms  
* Listings over time  
* Average price by property type  
* Geographic price comparisons  
* Correlation heatmaps  

---

## 🛠️ Feature Engineering

New features were created to improve model expressiveness:

* **Temporal features:** year, month, day_of_week  
* **One-hot encoding** for category, type, sub_type, state  
* **Binary indicators** for amenities  
* Capped numerical features while retaining raw features for anomaly detection  

---

## 🏷️ Price Fairness Labeling

Price fairness labels were created using **relative deviation from comparable listings**:

```python
price_diff_pct = (price - market_baseline) / market_baseline

## 🏷️ Labeling Rules

| Condition | Label |
|-----------|-------|
| diff < −0.15 | Underpriced |
| −0.15 ≤ diff ≤ +0.15 | Fair |
| diff > +0.15 | Overpriced |

> These thresholds balance class distribution, business realism, and model learnability.

---

## 🧩 Segmented Modeling Strategy

The real estate market has **two distinct regimes**:

| Segment | Characteristics |
|---------|----------------|
| Rental | Monthly pricing, high turnover, amenity-driven |
| Sale   | Capital pricing, long-term value, location-driven |

> Training separate models for each segment reduces noise and improves predictive stability.

---

## 📌 Models Trained

### 1️⃣ Baseline — Logistic Regression
* Transparent coefficients  
* Fast training  
* Strong benchmarking baseline  

### 2️⃣ Advanced — XGBoost & Random Forest
* Capture nonlinear interactions and complex location effects  
* Handle cross-feature dependencies  
* Well-suited for structured tabular real estate data  
* Random Forest provides a robust ensemble-based approach with feature importance insights, complementing XGBoost  

---

## 📈 Model Performance (Summary)

| Model | Segment | F1 Score | ROC AUC |
|-------|---------|----------|---------|
| Logistic Regression | Rental | High | High |
| Logistic Regression | Sale | High | High |
| XGBoost | Rental | Very High | Very High |
| XGBoost | Sale | Very High | Very High |
| Random Forest | Rental | Very High | Very High |
| Random Forest | Sale | Very High | Very High |

**Key Observations:**

* XGBoost and Random Forest outperform Logistic Regression  
* Segmented models outperform pooled models  
* Fair class is easiest to predict  
* Underpriced and Overpriced classes achieve strong recall  
* High ROC-AUC indicates excellent class separation  

---

## 📊 Visual Model Diagnostics

The notebook includes:

* Confusion matrices  
* ROC curves  
* Feature importance plots (XGBoost & Random Forest)  
* Coefficient plots (Logistic Regression)  

> These visualizations provide error analysis, interpretability, and business explainability.

---

## 🚀 Future Improvements

* Hyperparameter tuning  
* SHAP explainability  
* Geographic embeddings  
* Price anomaly scoring  
* Time-aware models  
* Real-time API deployment  

---

## 🧾 How to Run

```bash
pip install -r requirements.txt
jupyter notebook FINAL_NOTEBOOK.ipynb

