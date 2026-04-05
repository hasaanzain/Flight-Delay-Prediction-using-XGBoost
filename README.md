# Flight-Delay-Prediction-using-XGBoost
A machine learning project that predicts whether a flight will be delayed by more than 60 minutes using structured flight data and an optimized XGBoost classifier.

---

## Overview

This project builds an end-to-end pipeline for:

- Data preprocessing and feature engineering  
- Exploratory data analysis (EDA)  
- Binary classification modeling  
- Model evaluation using ROC-AUC and confusion matrix  
- Hyperparameter tuning with cross-validation  

The goal is to classify flights into:
- `0` → Not delayed (> 60 minutes)  
- `1` → Delayed (> 60 minutes)  

---


## 📂 Dataset

The dataset used was downloaded from kaggle and can be found here: https://www.kaggle.com/datasets/undersc0re/flight-delay-and-causes?resource=download

The cleaned dataframe includes the following features:

- `DayOfWeek`
- `Date`
- `DepTime`
- `Airline`
- `Origin`
- `Dest`
- `CarrierDelay`

### 🎯 Target Variable

```python
is_delayed_60+ = 1 if CarrierDelay > 60 else 0
```


## Tech Stack
- Python
- pandas
- NumPy
- scikit-learn
- xgboost
- matplotlib


## Installation

Clone the repository and install dependencies:

git clone https://github.com/yourusername/flight-delay-prediction.git
cd flight-delay-prediction
pip install -r requirements.txt

Or install manually:

pip install pandas numpy xgboost scikit-learn matplotlib seaborn


## Workflow
1. Data Preprocessing
Selected relevant columns
Converted Date to datetime
Extracted month and day
Removed original date column
2. Feature Engineering
One-hot encoded categorical variables:
Airline
Origin
Destination
3. Target Creation
df_encoded['is_delayed_60+'] = np.where(df_encoded['CarrierDelay'] > 60, 1, 0)
4. Train-Test Split
70% training
30% testing
5. Exploratory Data Analysis (EDA)
Delay percentage by airline
Delay trends by day of week
Delay distribution across origin airports
6. Model Training
Baseline XGBoost classifier
7. Model Evaluation
Accuracy
Confusion Matrix
ROC Curve
AUC Score
8. Hyperparameter Tuning

Used GridSearchCV to optimize:

learning_rate
max_depth
n_estimators
subsample

 
## Evaluation Metrics

The model is evaluated using:

- Accuracy → overall correctness
- Confusion Matrix → breakdown of predictions
- ROC Curve → model performance across thresholds
- AUC Score → probability that the model ranks positives higher than negatives

# Results
- Model	Accuracy	AUC
- Baseline XGBoost	~0.XX	~0.XX
- Tuned XGBoost	~0.XX	~0.XX

The baseline XGBoost model already captured the underlying patterns effectively. Given the feature space and dataset size, additional hyperparameter tuning resulted in minimal performance gains, suggesting the model was near its optimal capacity

📉 Visualizations

<p align="center">
  <img src="images/ROC_curve.png" width="45%" />
  <img src="images/histogram.png" width="45%" />
</p>


🧠 Key Learnings
Gradient boosting models like XGBoost are highly effective for tabular data
Feature engineering significantly impacts performance
ROC-AUC is more informative than accuracy for imbalanced datasets
Hyperparameter tuning can meaningfully improve results




👤 Author

Hasaan Mohsin

Physics & Astronomy, University of Toronto
Data Scientist & AI Engineer

