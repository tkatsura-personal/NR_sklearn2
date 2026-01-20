# -*- coding: utf-8 -*-
"""This used to be a file
"""

#Load sklearn kits: source sklearn-env/bin/activate
import numpy as np
import pandas as pd
import datetime as dt

from copy import deepcopy
from pyread import measurement, onset
from matplotlib import pyplot as plt
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Find duplicate entries
#duplicates = measurement[measurement.duplicated(subset=['NR_Name', 'week'], keep=False)]
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(duplicates)

# Change value of weeks or remove data point
measurement.loc[measurement['measurement_ID'] == 56925, 'week'] = 16
measurement.loc[measurement['measurement_ID'] == 56926, 'week'] = 16
measurement.loc[measurement['measurement_ID'] == 56277, 'week'] = 38
measurement.loc[measurement['measurement_ID'] == 56280, 'week'] = 38
measurement.loc[measurement['measurement_ID'] == 56367, 'week'] = 38

# Remove row for measurement_ID is not between 56134 and 56136
measurement = measurement[~measurement['measurement_ID'].isin([56035, 56472])]

# deepcopy measurement with only week < 12
weightData = deepcopy(measurement[measurement.week < 12][['NR_Name', 'week', 'weight']])
rbgData = deepcopy(measurement[measurement.week < 12][['NR_Name', 'week', 'rbg']])
# Pivot the dataframes
weightData = weightData.pivot(index='NR_Name', columns='week', values='weight')
rbgData = rbgData.pivot(index='NR_Name', columns='week', values='rbg')

# Reset index, rename columns, and merge the dataframes
weightData = weightData.reset_index()
rbgData = rbgData.reset_index()
weightData.columns = ['NR_Name'] + [f'wt{col}' for col in weightData.columns if isinstance(col, int)]
rbgData.columns = ['NR_Name'] + [f'rbg{col}' for col in rbgData.columns if isinstance(col, int)]
dataset = pd.merge(weightData, rbgData, on='NR_Name', how='outer')

# Merge with onset where overall_diet == 'Rod' on NR_Name. Include sex, gestational_diet, nursing_diet, rbg200.
dataset = pd.merge(dataset, 
                   onset[onset['overall_diet'] == 'Rod'][['NR_Name', 'sex', 'gestational_diet', 'nursing_diet', 'rbg200']],
                    on='NR_Name', how='left')

# Drop rows with missing values
dataset.dropna(inplace=True)

# Create a DataFrame to hold mean CV scores for sex per column
mean_cv_scores = pd.DataFrame(columns=['Model', 'Mean CV Score', 'Sex'])
std_cv_scores = pd.DataFrame(columns=['Model', 'STD CV Score', 'Sex'])

for sex in ['M', 'F']:
    
    if sex == 'M':
        week_cutoff = 16
    else:
        week_cutoff = 36
    
    # Filter by sex
    filtered = dataset[dataset['sex'] == sex]
    # Change rbg200 such that true if less than week_cutoff
    filtered['rbg200'] = filtered['rbg200'] <= week_cutoff
    # Change diet so that Rod is 1 and everything else is 0
    filtered['gestational_diet'] = (filtered['gestational_diet'] == 'Rod').astype(int)
    filtered['nursing_diet'] = (filtered['nursing_diet'] == 'Rod').astype(int)

    # Setup sklearn 5 fold validation for Logistic, Ridge, LASSO, KNN, RF, DT, and NN
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Ridge Regression': RidgeClassifier(max_iter=500),
        'LASSO Regression': LogisticRegression(max_iter=500, solver='liblinear', penalty='l1'),
        'KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Neural Network': MLPClassifier(max_iter=1000)
    }

    # Prepare data
    X = filtered.drop(columns=['NR_Name', 'sex', 'rbg200'])
    y = filtered['rbg200']

    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        cv_results = cross_val_score(model, X, y, cv=rkf)
        print(f"CV Results (first 10 shown): {cv_results[:10]}")
        print(f"Mean CV Score: {cv_results.mean():.4f}")
        print(f"Std CV Score: {cv_results.std():.4f}\n")
        # Add the mean cv to dataframe, mean_cv_scores, with first column for male, 2nd column for female
        if sex == 'M':
            mean_cv_scores = mean_cv_scores._append({'Model': model_name, 'Mean CV Score': cv_results.mean(), 'Sex': 'M'}, ignore_index=True)
            std_cv_scores = std_cv_scores._append({'Model': model_name, 'STD CV Score': cv_results.std(), 'Sex': 'M'}, ignore_index=True)
        else:
            mean_cv_scores = mean_cv_scores._append({'Model': model_name, 'Mean CV Score': cv_results.mean(), 'Sex': 'F'}, ignore_index=True)
            std_cv_scores = std_cv_scores._append({'Model': model_name, 'STD CV Score': cv_results.std(), 'Sex': 'F'}, ignore_index=True)

# widen mean_cv_scores
mean_cv_scores_pivot = mean_cv_scores.pivot(index='Model', columns='Sex', values='Mean CV Score')
# Plot the mean cv scores vertically and dodge
x = np.arange(len(mean_cv_scores_pivot.index))  # positions
width = 0.35  # bar width

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, mean_cv_scores_pivot['M'], width, color='skyblue', label='Male')
plt.bar(x + width/2, mean_cv_scores_pivot['F'], width, color='lightcoral', label='Female')

plt.xticks(x, mean_cv_scores_pivot.index, rotation=45)
plt.xlabel('Model Type')
plt.ylabel('Mean CV Score')
plt.title('Model Comparison')
plt.tight_layout()
plt.ylim(0.7, 1)
plt.show()

# widen std_cv_scores
std_cv_scores_pivot = std_cv_scores.pivot(index='Model', columns='Sex', values='STD CV Score')
# Plot the std cv scores vertically and dodge
x = np.arange(len(std_cv_scores_pivot.index))  # positions
width = 0.35  # bar width
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, std_cv_scores_pivot['M'], width, color='skyblue', label='Male')
plt.bar(x + width/2, std_cv_scores_pivot['F'], width, color='lightcoral', label='Female')

plt.xticks(x, std_cv_scores_pivot.index, rotation=45)
plt.xlabel('Model Type')
plt.ylabel('STD CV Score')
plt.title('Model Comparison')
plt.tight_layout()
plt.ylim(0.0, 0.1)
plt.show()
