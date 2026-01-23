# 🏠 FairPrice: Real Estate Price Fairness Classification

---

## 📌 Project Overview

The real estate market suffers from **pricing opacity**. Buyers, renters, sellers, and agents often rely on intuition or incomplete comparables to judge whether a property is reasonably priced.  

To address this, we built a **machine learning classification system** that labels property listings as:

* **Underpriced**  
* **Fairly Priced**  
* **Overpriced**  

Unlike traditional price prediction (a noisy regression task), this system focuses on **relative price fairness**, producing outputs that are **interpretable, actionable, and aligned with real-world decision making**.

---
## 🧰 Business Problem

Pricing inefficiencies in real estate create risk and lost value for all participants:

* Buyers and renters may overpay due to poor market transparency
* Sellers and landlords may underprice assets and lose potential revenue
* Agents face difficulty justifying prices without objective benchmarks

The core business question addressed in this project is:

> *Can we use historical real estate data and property features to reliably classify listings as underpriced, fairly priced, or overpriced?*

---

## 🎯 Business Objectives

The project aims to improve pricing transparency in the Kenyan real estate market by:
  
* Provide buyers and renters with a simple, interpretable indicator of whether a listing price is reasonable
* Support sellers and landlords in identifying potential underpricing and revenue leakage
* Equip agents and analysts with a data-driven pricing benchmark to improve negotiation and valuation accuracy
* Reduce market inefficiencies caused by information asymmetry
  
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

The cleaned and feature-engineered dataset contains **9,091 property listings** after cleaning with fields such as:

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
  <img width="849" height="545" alt="image" src="https://github.com/user-attachments/assets/ae8b296d-2827-4d0b-ba44-13ab06689837" />

* Listings over time
  <img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/62f21385-033d-4eec-80cf-c2f207d8a19f" />

* Average price by property type
  <img width="1004" height="645" alt="image" src="https://github.com/user-attachments/assets/d4f1f664-d186-4c9a-9040-0b3abff8847b" />

* Geographic price comparisons
  <img width="1004" height="590" alt="image" src="https://github.com/user-attachments/assets/3df10cfd-c39d-4b39-83e8-74df08b2606d" />
  <img width="1004" height="593" alt="image" src="https://github.com/user-attachments/assets/32bf6f2d-b352-4c45-9446-1f861f0b7f8f" />

* Correlation heatmaps
  <img width="825" height="740" alt="image" src="https://github.com/user-attachments/assets/a2a4017c-6f56-4046-bbef-204b4491c99b" />


---

## 🛠️ Feature Engineering

Key transformations included:

* Creation of a **price fairness label** (`price_label`) based on market-relative pricing logic
* Extraction of temporal features:

  * `year`, `month`, `day_of_week`
* One-hot encoding of categorical variables such as:

  * Property type and subtype
  * State and other location indicators
* Boolean normalization for features like:

  * Furnished, serviced, shared, parking

The final feature matrix excludes the target variable (`price_label`) to prevent leakage. 

---

## 📌 Models Trained

### 1️⃣ Baseline — Logistic Regression
Used as a linear baseline with class weighting to handle imbalance.  

### 2️⃣ Advanced — Tree-Based Models
* **Random Forest Classifier**
  Chosen for:

  * Nonlinear decision boundaries
  * Robustness to outliers
  * Interpretability via feature importance

* **XGBoost Classifier**
  Configured for multiclass classification using `multi:softprob` with:

  * Moderate tree depth
  * Learning rate control for stability
  * Subsampling and column sampling to reduce overfitting


## 📈 Model Performance (Summary)

|* **Logistic Regression**

  * Served as a strong linear baseline
  * Struggled with nonlinear relationships and complex interactions

* **Random Forest**

  * Improved performance across all classes
  * Demonstrated strong generalization
  * Provided stable feature importance rankings

* **XGBoost** (Best Performing Model)

  * Achieved the highest overall F1-score
  * Showed the best balance between precision and recall
  * Handled class imbalance effectively

The final production candidate is the **XGBoost multiclass classifier**.

## ⚙️ Deployment Considerations

The trained model was serialized as part of a full preprocessing + modeling pipeline.
* Model file size is relatively large due to:

  * High number of trees
  * One-hot encoded categorical features

* Slow load times were observed during deployment.
* Deployment link: https://fmk4v2pivccvxcycxynqqf.streamlit.app/

**Mitigation Strategies**

* Reduced number of estimators
* Considered tree depth pruning
* Evaluated model compression options
* Ensured only the final trained pipeline is saved

---

## 🚨 Limitations

* Labels are market-relative and depend on historical pricing patterns
* Short-term rental dynamics are excluded
* Extreme luxury properties are underrepresented
* Geospatial granularity is limited to categorical location features

---

## ✅ Business Recommendations

Based on the model outputs and exploratory analysis, the following business recommendations are proposed:

* **Adopt FairPrice Check as a decision-support tool**
  Real estate platforms, agencies, and property developers can integrate the model to flag listings that are likely overpriced or underpriced before publication.

* **Use classification labels to guide pricing strategy**

  * Overpriced listings can be reviewed and adjusted to reduce time-on-market
  * Underpriced listings can be repriced to capture lost revenue
  * Fairly priced listings can be fast-tracked for marketing and promotion

* **Segment pricing strategies by sale/rent, location and property type** 
  EDA shows that pricing behavior varies significantly by sale/rent, region and property subtype. Localized pricing rules should be layered on top of the model outputs.

* **Leverage explainability for stakeholder trust**
  Feature importance and future SHAP explanations should be used to justify model-driven recommendations to clients and internal teams.
  
---

## ♻️ Future Work

* Incorporate geospatial coordinates and neighborhood-level features
* Integrate macroeconomic indicators
* Expand dataset with more recent listings
* Explore deep learning for feature representation
* Add SHAP-based explainability for end users

---

## 📶 Repository Structure

```
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── FINAL_NOTEBOOK.ipynb
├── models/
│   └── xgboost_pipeline.pkl
├── app/
│   └── app.py
├── README.md
```

---

## ⏸️ Conclusion

FairPrice Check demonstrates how supervised machine learning can be used to transform noisy real estate listing data into a practical, interpretable pricing fairness tool. By framing price assessment as a classification task rather than a regression problem, the system provides actionable guidance that is easier for end users to trust and apply in real-world decision-making.

## 🧑‍🧑‍🧒‍🧒 Collaborators
* Jeremiah Bii                            
* Faith Kanyuki
* Michelle Ngunya
* Kelvin Omina
* Michelle Maina
* Faith Kanyuki
* Fatuma Tari

