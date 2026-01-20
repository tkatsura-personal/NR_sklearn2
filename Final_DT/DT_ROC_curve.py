# -*- coding: utf-8 -*-
"""This used to be a file
"""

#Load sklearn kits: source sklearn-env/bin/activate
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn import tree
from sklearn.metrics import auc, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime


#Input week to cutoff for diabetes
week_cutoff = int(input("Enter week cutoff: "))

#feature_sub will be the features use includeParentOnset to add the parent onset as inputs
#Do not include father and mother onset.
print("Insert features to include, separated by comma (,) no spaces.")
print("Available features are gestational_diet,nursing_diet,wt4,wt8,wt12,rbg4,rbg8,rbg12")
feature_input = input("Insert features: ")
feature_sub = list(feature_input.split(","))
#['gestational_diet', 'nursing_diet', 'wt4', 'wt8', 'wt12', 'rbg4', 'rbg8', 'rbg12'] 
#Features to run Decision Tree on, All possible features are below:
#gestational_diet, nursing_diet, wt4, wt8, wt12, rbg4, rbg8, rbg12, rbg200_fa, rbg200_mo'
#For copy-pasting, ['gestational_diet', 'nursing_diet', 'wt4', 'wt8', 'wt12', 'rbg4', 'rbg8', 'rbg12'] 
includeFamilyOnset = input("Consider family onset together? (y or n): ").lower() == "y"
if not includeFamilyOnset:
    includeParentOnset = input("Include parent onset? (y or n): ").lower() == "y" #Include parent onset as a feature
    includeSiblingOnset = input("Include sibling onset? (y or n): ").lower() == "y" #Include sibling onset as a feature. This includes half-siblings.
else:
    includeParentOnset = True
    includeSiblingOnset = True

#Read from file
R_measure = pd.read_csv('measurement.csv', header = 0, index_col = 0)
R_onset_all = pd.read_csv('onset.csv', header = 0, index_col = 0)

#Remove rats that did not get diabetes and got euthanized before 28 weeks
R_onset_all['birth_date'] = pd.to_datetime(R_onset_all['birth_date'])
R_onset_all['death_date'] = pd.to_datetime(R_onset_all['death_date'])

R_onset = deepcopy(R_onset_all[(R_onset_all.overall_diet == "Rod") & 
                               (((R_onset_all['death_date'] - R_onset_all['birth_date']).dt.days >= 7*week_cutoff) | 
                                R_onset_all['rbg200'] > 0)]) #Lived 28+ weeks or gotten rbg200

#Remove unnecessary rows (measurements pas 12 weeks)
measure_sample = R_measure[(R_measure.week < 16) & (R_measure.week % 4 == 0)][['NR_Name', 'week', 'weight', 'weight percentile', 'rbg']]

#Pivot the dataframe and remove na values for 4 and 8 week weight.
wt = measure_sample.pivot(index = 'NR_Name', columns = 'week', values = 'weight')

#Similar thing to rbg values, but also remove any rats above 200 (cutoff for diabetes)
rbg = measure_sample.pivot(index = 'NR_Name', columns = 'week', values = 'rbg')

#Merge all dataframes and rename relevant columns.
recording_all = pd.merge(wt, rbg, on = 'NR_Name').rename(
    columns={'4_x':'wt4', '8_x':'wt8', '12_x':'wt12', '4_y':'rbg4', '8_y':'rbg8', '12_y':'rbg12'}
    )

#Make non-numerical values into a boolean, 1 are the ones that tends to get diabetic
R_onset['nursing_diet'] = R_onset['nursing_diet'].isin(['Rod', 'New'])
R_onset['gestational_diet'] = R_onset['gestational_diet'].isin(['Rod', 'New'])
R_onset['weanling_diet'] = R_onset['weanling_diet'].isin(['Rod', 'New'])

#Merges with mother and father onset time
allData = pd.merge(recording_all[['wt4', 'wt8', 'wt12', 'rbg4', 'rbg8', 'rbg12']],
                   R_onset[['NR_Name', 'mother', 'father', 'sex', 'gestational_diet', 'nursing_diet', 'weanling_diet', 'rbg200']], 
                   on = "NR_Name")

allData['diff48'] = allData['wt8'] - allData['wt4']

allData_mother = pd.merge(allData, R_onset_all[['NR_Name', 'rbg200']], how = 'left', left_on='mother', right_on='NR_Name').rename(
    columns = {'rbg200_x':'rbg200', 'rbg200_y': 'rbg200_mo', 'NR_Name_x':'NR_Name', 'NR_name_y':'mother'})

allData_parent = pd.merge(allData_mother, R_onset_all[['NR_Name', 'rbg200']], how = 'left', left_on='father', right_on='NR_Name').rename(
    columns = {'rbg200_x':'rbg200', 'rbg200_y': 'rbg200_fa', 'NR_Name_x':'NR_Name', 'NR_name_y':'father'})

parent_to_children = {}

allData_parent['isOnset'] = allData_parent['rbg200'].apply(lambda x: 1 if pd.notna(x) and int(x) <= week_cutoff else 0)

# Function to check if any sibling has a degree
def has_sibling_with_degree(person_id):
    siblings = set()
    
    # Find siblings by mother and father
    for parent in ['mother']:
        parent_id = allData_parent.loc[allData_parent['NR_Name'] == person_id, parent].values[0]
        if pd.notna(parent_id) and parent_id in parent_to_children:
            siblings.update(parent_to_children[parent_id])
    
    # Remove self from sibling list
    siblings.discard(person_id)
    
    # Check if any sibling has onset
    return int(allData_parent[allData_parent['NR_Name'].isin(siblings)]['isOnset'].any())

# Apply function to each row
if (includeSiblingOnset):
    for _, row in allData_parent.iterrows():
        for parent in ['mother', 'father']:
            if pd.notna(row[parent]):  # Ignore missing parent IDs
                parent_to_children.setdefault(row[parent], []).append(row['NR_Name'])

    allData_parent['rbg200_sib'] = allData_parent['NR_Name'].apply(has_sibling_with_degree)

#column_to_move will be the features, removed wt12 and rbg
column_to_keep = 'rbg200'
features = deepcopy(feature_sub)
if (includeParentOnset):
    features += ['rbg200_fa', 'rbg200_mo']
if (includeSiblingOnset):
    features += ['rbg200_sib']
noNA = allData_parent[features + [column_to_keep] + ['sex']].dropna(subset = feature_sub)
if (includeParentOnset):
    noNA['rbg200_fa'] = noNA['rbg200_fa'] <= week_cutoff
    noNA['rbg200_mo'] = noNA['rbg200_mo'] <= week_cutoff
if (includeFamilyOnset):
    noNA['family_history'] = noNA['rbg200_fa'] | noNA['rbg200_mo'] | noNA['rbg200_sib']
    features.remove('rbg200_fa')
    features.remove('rbg200_mo')
    features.remove('rbg200_sib')
    features += ['family_history']
    noNA.drop(['rbg200_fa', 'rbg200_mo', 'rbg200_sib'], axis = 1)

for gender in ['M', 'F']:
    noNA_sex = noNA[(noNA['sex'] == gender) & (noNA['rbg8'] < 200)].drop('sex', axis = 1)
    print(f"There are {len(noNA_sex)} rats")
    #Create x and y train and test set to use for 5-fold CV
    x = noNA_sex[features] 
    x = x.fillna(80) #This is to fill parent onset weeks, it's set to 80 but becomes a boolean later
    
    y = noNA_sex[column_to_keep]
    y = y.fillna(80) #This will just become a boolean so it just needs to be bigger than week_cutoff
    y = y <= week_cutoff
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=888)

    #5-fold CV for Deicision Tree, using Decision Tree because 5-fold CV tunes the hyperparameters.
    sklearn_5fold_DT = GridSearchCV(
        DecisionTreeClassifier(),
        param_grid={
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [3, 4, 5, 6]
        },
        cv=5,
        scoring='accuracy'
    )
    sklearn_5fold_DT.fit(x_train, y_train)
    print(sklearn_5fold_DT.best_params_)

    #Use the parameters from 5-fold CV for the sample
    sklearn_DT = DecisionTreeClassifier(splitter = sklearn_5fold_DT.best_params_['splitter'], 
                                    criterion = sklearn_5fold_DT.best_params_['criterion'],
                                    max_depth = sklearn_5fold_DT.best_params_['max_depth']) #Insert best parameters
    sklearn_DT.fit(x_train, y_train)

    #Print out which features are important, as well as showing what the decision tree looks like.
    print(sklearn_DT.feature_importances_)
    y_pred = sklearn_DT.predict(x_test)
    text_representation = tree.export_text(sklearn_DT)
    print(text_representation)
    print(classification_report(y_test, y_pred))

    #Plots the ROC curve of the model.
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    print(f'model 1 AUC score: {roc_auc_score(y_test, y_pred)}')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show() #Plot shown will need to be closed before the next can be opened

    #Plots the ROC curve for each feature
    plt.figure(figsize=(10, 8))
    for i in range(x.shape[1]):
        # Get the probabilities for class 1
        y_probs = sklearn_DT.predict_proba(x_test)[:, 1]
        
        # Calculate the ROC curve and AUC for each feature
        fpr, tpr, thresholds = roc_curve(y_test, x_test.iloc[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{features[i]} (AUC = {roc_auc:.2f})')

    # Plot random chance (diagonal line)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')

    # Customize the plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Feature (Week ' + str(week_cutoff) + ', ' + gender + ")")
    plt.legend(loc='lower right')
    plt.show()