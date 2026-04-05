# Generated from: EDA.ipynb
# Converted at: 2026-04-05T06:39:29.887Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Airline Delay Predictor Algorithm using XGBoost
# ## Author: Hasaan Mohsin


# import necessary libraries

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

#load in dataframe and display first 5 rows

df = pd.read_csv('Flight_delay.csv')
df.head()

df.shape

# We do not need all the columns for the model, so we will drop the ones we do not need, and only keep the following: 

df = df[['DayOfWeek','Date','DepTime','Airline','Origin','Dest','CarrierDelay']]

# Check null values

df.isnull().sum()



# display top 5 rows again

df.head()

# okay so i can see that i should probably convert the date to a datetime object

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

#also seperating month and day into their own columns
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

#can safely drop the original date column now
df = df.drop(columns=['Date'])

df.head()

# XGBoost won't allow for categorical variables, so we need to convert them to numerical variables
categories = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=['Airline', 'Origin', 'Dest'], drop_first=True)

# display top 5 rows again

df_encoded.head()

# TARGET VARIABLE:

df_encoded['is_delayed_60+'] = np.where(df_encoded['CarrierDelay'] > 60, 1, 0)

# define, split and train the model

X = df_encoded.drop(columns=['is_delayed_60+', 'CarrierDelay'])
y = df_encoded['is_delayed_60+']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

train_set = pd.concat([X_train, y_train], axis= 1)

# Reversing one-hot encoding for multiple sets of categorical variables
for category in categories:
    one_hot_columns = [col for col in train_set.columns if col.startswith(f'{category}_')]
    train_set[category] = train_set[one_hot_columns].idxmax(axis=1)
    train_set = train_set.drop(columns=one_hot_columns)
    train_set[category] = train_set[category].str.replace(f'{category}_', '')

train_set

# check distribution of target variable

train_set['is_delayed_60+'].value_counts()

# By airline

train_set.groupby('Airline')['is_delayed_60+'].mean().sort_values(ascending=False).round(3)*100

# By day of week:

DayOfWeek_pct_delayed = train_set.groupby('DayOfWeek')['is_delayed_60+'].mean().round(3)*100
DayOfWeek_pct_delayed

# By origin:

pct_delay_by_origin = train_set.groupby('Origin')['is_delayed_60+'].mean().sort_values(ascending=False).round(3)*100
pct_delay_by_origin.head(20)

# Histogram

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(pct_delay_by_origin.values, bins=25, color='blue', edgecolor='black')

# Add labels and title
plt.title("Distribution of 60+ Minute Delays By Origins", fontsize=14)
plt.xlabel("Percentage of 60+ Minute Delays (%)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

# Show the plot
plt.show()

#initialize and fit the XGBoost classifier

xgb_model = xgb.XGBClassifier(random_state=0, 
                              eval_metric='logloss')

xgb_model 

xgb_model.fit(X_train, y_train)
# make predictions on the test set

y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
# evaluate the model


print("XGBoost Classifier (Baseline):")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Confusion Matrix:

cm = confusion_matrix(y_test, y_pred)
cm

#Caluclate AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score:.4f}")

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# DO CROSS VALIDATED GRID SEARCH

# Define the parameter grid
param_grid = {
    'learning_rate': [0.01, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 250],
    'subsample': [0.6,  1.0]
}

# Initialize the XGBoost model
xgb_model = xgb.XGBClassifier(random_state=0,
                              eval_metric='logloss')

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, 
                           param_grid=param_grid, 
                           cv=3, 
                           scoring='roc_auc', 
                           verbose=1, 
                           n_jobs=-1)

# Fit the grid search model
grid_search.fit(X_train, y_train)


# Evaluate best parameters and predict on training set


# Best parameters from GridSearch
print("Best parameters found: ", grid_search.best_params_)

# Predict with the best model
y_pred_best = grid_search.best_estimator_.predict(X_test)

# Evaluate the tuned XGBoost model
print("XGBoost Classifier (Tuned):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
cm

# Predict probabilities for the test set (to calculate AUC)
y_pred_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]  # We need probabilities for the positive class

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score:.4f}")

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()