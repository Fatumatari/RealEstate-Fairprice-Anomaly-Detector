🏠 FairPrice: Real Estate Price Fairness Classification

A Machine Learning System for Detecting Underpriced, Fairly Priced, and Overpriced Listings

📌 Project Overview

The real estate market suffers from pricing opacity. Buyers, renters, sellers, and agents often rely on intuition or incomplete comparables to judge whether a property is reasonably priced. This leads to:

Overpriced listings staying too long on the market

Underpriced listings causing seller losses

Poor negotiation outcomes

Reduced trust in listing platforms

This project builds a machine learning classification system that labels property listings as:

Underpriced

Fairly Priced

Overpriced

Instead of predicting exact prices (a noisy regression task), this system focuses on relative price fairness, producing outputs that are interpretable, actionable, and aligned with real-world decision making.

🎯 Business Objectives

Classify property listings into underpriced, fair, or overpriced categories

Support better pricing decisions for buyers, renters, and sellers

Improve pricing transparency in real estate markets

Enable scalable pricing analysis across cities and property types

Provide interpretable model outputs for non-technical users

🧠 Why Classification Instead of Regression?

Traditional price prediction models are:

Highly sensitive to outliers

Unstable in heterogeneous markets

Hard to interpret for end users

This project reframes pricing as a multi-class classification problem, which:

Is more robust to noise

Aligns with real-world decision logic

Produces directly actionable labels

Improves interpretability

📂 Dataset Description

The cleaned and feature-engineered dataset contains 9,091 property listings with the following structure:

Key Fields

price (capped for modeling stability)

price_raw (original uncapped value)

bedrooms, bathrooms, toilets, parking

furnished, serviced, shared

category (For Rent / For Sale)

type, sub_type

state, locality

listdate

Capping Rules Applied

Feature	Lower Bound	Upper Bound
Bedrooms	0	10
Bathrooms	0	10
Parking	0	10
Rent Price	5,500	600,000
Sale Price	600,000	100,000,000

The raw versions (*_raw) were retained for anomaly detection and interpretability.

🔍 Exploratory Data Analysis (EDA)

EDA was conducted to understand market structure, pricing dynamics, and feature relationships.

Key Analyses

Distribution of prices (rent vs sale)

Price trends over time

Property types and sub-types

Geographic price differences (state & locality)

Feature distributions (bedrooms, bathrooms, amenities)

Correlation heatmaps

Category-wise comparisons

Key Visualizations

📊 Price distribution histograms

📈 Listings over time

🏘️ Average price by property type

🌍 Geographic price comparisons

🔥 Correlation heatmaps

These visuals revealed:

Heavy price skewness

Strong location effects

Nonlinear relationships

Distinct rental vs sale market regimes

🛠️ Feature Engineering

New features were engineered to improve model expressiveness:

Temporal features: year, month, day_of_week

One-hot encoding for:

category

type

sub_type

state

Binary indicators for amenities

Capped numerical features

Retained raw features for anomaly detection

🏷️ Price Fairness Labeling

Price fairness labels were created using relative deviation from comparable listings:

price_diff_pct = (price - market_baseline) / market_baseline


Labeling Rule

Condition	Label
diff < −0.15	Underpriced
−0.15 ≤ diff ≤ +0.15	Fair
diff > +0.15	Overpriced

This threshold was chosen to balance:

Class distribution

Business realism

Model learnability

🧩 Segmented Modeling Strategy

The real estate market operates under two structurally distinct regimes:

Segment	Characteristics
Rental	Monthly pricing, high turnover, amenity-driven
Sale	Capital pricing, long-term value, location-driven

Training separate models reduces cross-market noise and improves predictive stability.

📌 Models Trained
1️⃣ Baseline — Logistic Regression

Used as a transparent reference model:

Interpretable coefficients

Fast training

Clear bias–variance tradeoffs

Strong benchmarking baseline

2️⃣ Advanced — XGBoost

Used to capture:

Nonlinear interactions

Feature thresholds

Complex location effects

Cross-feature dependencies

XGBoost is well-suited for structured tabular real estate data.

📈 Model Performance (Summary)
Model	Segment	F1 Score	ROC AUC
Logistic Regression	Rental	High	High
Logistic Regression	Sale	High	High
XGBoost	Rental	Very High	Very High
XGBoost	Sale	Very High	Very High

Key Observations

XGBoost consistently outperforms Logistic Regression

Segmented models outperform pooled models

Fair class is easiest to predict

Underpriced and Overpriced classes achieve strong recall

High ROC-AUC indicates excellent class separation

📊 Visual Model Diagnostics

The notebook includes:

Confusion matrices

ROC curves

Feature importance plots

Coefficient plots (Logistic Regression)

These visualizations provide:

Error analysis

Interpretability

Model trustworthiness

Business explainability

💼 Business Value
Buyers & Renters

Identify good deals

Avoid overpriced listings

Improve negotiation leverage

Reduce financial risk

Sellers & Property Owners

Set competitive prices

Reduce time on market

Increase transaction success

Agents & Platforms

Add objective pricing indicators

Improve user trust

Differentiate with intelligent insights

🧠 Key Design Decisions
Decision	Rationale
Classification framing	More interpretable and robust than regression
Feature capping	Reduces outlier influence
Retaining raw features	Enables anomaly detection
Segmented modeling	Reduces regime noise
Logistic baseline	Interpretability
XGBoost advanced	Nonlinear learning
🚀 Future Improvements

Hyperparameter tuning

SHAP explainability

Geographic embeddings

Price anomaly scoring

Time-aware models

Real-time API deployment
