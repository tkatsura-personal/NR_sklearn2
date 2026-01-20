# -*- coding: utf-8 -*-
"""This used to be a file
"""

#Load sklearn kits: source sklearn-env/bin/activate
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
from copy import deepcopy
from pyread import measurement, onset
from matplotlib import pyplot as plt

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

# deepcopy measurement with only week <= 12
weightData = deepcopy(measurement[measurement.week <= 12][['NR_Name', 'week', 'weight']])
rbgData = deepcopy(measurement[measurement.week <= 12][['NR_Name', 'week', 'rbg']])
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

for sex in ['M', 'F']:
    # Remove NR_Name
    filtered = dataset[dataset['sex'] == sex].drop(columns=['NR_Name', 'sex'])
    # Convert diet to 1 or 0
    filtered['gestational_diet'] = (filtered['gestational_diet'] == 'Rod').astype(int)
    filtered['nursing_diet'] = (filtered['nursing_diet'] == 'Rod').astype(int)
    # Create a heatmap for the filtered data
    plt.figure(figsize=(10, 6))
    sns.heatmap(filtered.corr(), annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title(f'Correlation Heatmap - {sex}')
    plt.show()