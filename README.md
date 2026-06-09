# Prediction of the Total Cost of Studying Abroad

An advanced machine learning framework designed to predict the comprehensive cost of international education using structural academic and economic indicators. This project transitions analytical data findings into the **EduCost Card Builder**, a specialized interface built for education agencies to streamline personalized study-abroad cost estimations.

## 📌 Project Overview
Estimating the true financial commitment of studying abroad is highly challenging, with surveys indicating that only about 55% of students understand the real financial requirements beyond baseline tuition fees. This study implements data preprocessing pipelines, exploratory data analysis (EDA), and optimized boosting models to map, evaluate, and predict comprehensive global cost scenarios.

The project models expenses using a dual-framework:
1. **Univariate Modeling:** Predicts the consolidated aggregate total cost.
2. **Multivariate Modeling:** Concurrently breaks down specific expense categories (Tuition, Rent, Insurance, and Visa Fees).

---

## 📊 Dataset Specifications
The model utilizes the **Cost of International Education** dataset containing 907 observations across 12 distinct analytical dimensions:

| Variable Name | Type | Description |
| :--- | :--- | :--- |
| **Country** | Qualitative (Nominal) | ISO country name (comprises 71 unique countries) |
| **City** | Qualitative (Nominal) | City where the institution is located (56 unique cities) |
| **University** | Qualitative (Nominal) | Official name of the higher-education institution (622 unique institutions) |
| **Program** | Qualitative (Nominal) | Course/degree specific name (92 distinct programs) |
| **Level** | Qualitative (Ordinal) | Program level (`Bachelor`, `Master`, `PhD`) |
| **Duration Years** | Quantitative (Discrete) | Length of the academic program in years |
| **Tuition USD** | Quantitative (Continuous) | Total tuition cost across the full duration converted to USD |
| **Living Cost Index**| Quantitative (Continuous) | Daily living expense index benchmarked against New York City ($100$) |
| **Rent USD** | Quantitative (Continuous) | Average monthly accommodation rent in USD |
| **Visa Fee USD** | Quantitative (Continuous) | One-time international student visa application fee in USD |
| **Insurance USD** | Quantitative (Continuous) | Mandatory annual health/student insurance cost in USD |
| **Exchange Rate** | Quantitative (Continuous) | Local currency units per 1 U.S. Dollar |

* **Data Source:** [Kaggle - Cost of International Education Dataset](https://www.kaggle.com/datasets/adilshamim8/cost-of-international-education)

---

## 🛠️ Pipeline Architecture & Methodology

### 1. Feature Engineering
A foundational target variable (`Total_Cost_USD`) was structured mathematically to synthesize all structural dimensions into one total projection over the program's lifecycle:

$$\text{Total Cost (USD)} = \text{Tuition} + \left(\frac{\text{Living Cost Index}}{100} \times 12000 \times \text{Duration}\right) + (\text{Rent} \times 12 \times \text{Duration}) + \text{Visa Fee} + (\text{Insurance} \times \text{Duration})$$

### 2. Analytical Findings & Structural Insights
* **Multicollinearity Screening:** Evaluated via generalized variance inflation factors (GVIF) and correlation maps—confirming no crippling collinear dependency structures between structural predictors like `Duration`, `Exchange Rate`, and `Living Cost Index`.
* **Outlier Strategy:** Univariate analysis flagged massive real-world extremes (e.g., `Exchange Rate` had an outlier rate of 26.66%). Multivariate profiling using **Mahalanobis Distance** isolated 32 critical edge-case records ($\approx 5.05\%$). Rather than purging them—which degraded performance ($R^2$ dropped)—all outliers were retained to ensure the model captured premium universities and extreme economic regional tiers accurately.

### 3. Model Progression & Optimization
The system was evaluated through iterative steps using a 70% Train / 30% Test split across multiple regression models with 5-fold cross-validation:
* **Stage 1:** Living Cost Index bound inside the response function (Random Forest achieved a Baseline Test $R^2 = 0.9269$).
* **Stage 2:** Extracting the raw `Living Cost Index` to act as an independent standalone predictor feature (Gradient Boosting pushed Test $R^2 \rightarrow 0.9568$).
* **Stage 3 (Feature Selection):** Leveraged **SHAP (SHapley Additive exPlanations)** to isolate key driving variables (`Country`, `Living Cost Index`, and `Exchange Rate`), discarding lower-influence features to eliminate overfitting.

---

## 🚀 Final Performance & Best Model
The final system adopts a **Multivariate CatBoost** configuration which yielded superior prediction capabilities across all testing splits:

* **Univariate/Multivariate CatBoost Core Performance:**
  * **Test $R^2$ Score:** $\approx 0.9483$ (Explains nearly 95% of total real-world data variance)
  * **Test RMSE:** $\$5,226.86$
* **Category Breakdown Capabilities:** Achieved near-perfect predictive alignment ($\ge 0.99$) for static categories like Visa fees while heavily outperforming standard ElasticNet models on complex volatile vectors like Rent, Tuition, and Insurance.

---

## 📦 Core Product: EduCost Card Builder
The analytical modeling yields the **EduCost Card Builder**—an interactive interface using a dual-engine architecture:
1. **Univariate Engine:** Employs a specific CatBoost framework to generate instantaneous macro-level estimates of total journey costs.
2. **Multivariate Engine:** Dynamically calculates an itemized line-by-line financial breakdown across the core cost domains (Tuition, Rent, Insurance, and Visa) to give students an completely transparent financial layout.

---

## 💻 Technical Stack
* **Language:** Python 3.x
* **Core Libraries:** `CatBoost`, `XGBoost`, `scikit-learn`, `SHAP`
* **Data Processing & Analytics:** `pandas`, `numpy`, `scipy`
* **Visualization:** `matplotlib`, `seaborn`
* **Environment:** Google Colab / Jupyter Notebooks

---

## 👥 Group 05 Members
* **Vidura Gunawardana** (s16655)
* **Dinusha Priyashan** (s16798)
* **Manushi Pathirana** (s16859)

*Prepared for Course Units: DS 3003 / ST 3082 — Data Analysis Project 02 (Advanced Analysis)*

---

## 🔗 Project Links
* **Interactive Code Notebook:** [Google Colab Notebook](https://colab.research.google.com/drive/1aPPx0YRzOCfdLxBHXKu2Y-wk4unRn0-x?usp=sharing#scrollTo=cfrzGuCE5ffh)
