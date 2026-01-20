
# This code is me attempting to fix the issues that jointModel.R had with the C-index calculation
# The original code used the survcomp package, which is no longer maintained and has compatibility issues with recent R versions.
# Additionally, the analysis was slow and was not able to optimize for certain scoring methods.
# Although python does not have a direct equivalent to the JMbayes2 package in R, there is a system where
# I can create my own joint models using libraries like lifelines for survival analysis and statsmodels for longitudinal data.

# Load necessary packages
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import pymc as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer

# Load the dataset
data = pd.read_csv("onset_longitudinal.csv")

# Preprocess the data
data = data.dropna()
data['NR_Name'] = data['NR_Name'].astype('category')
data['sex'] = data['sex'].astype('category')
data['diet'] = data['diet'].astype('category')
data['Event'] = data['Event'].astype(int)
data['Event_time'] = data['Event_time'].astype(float)
data['week'] = data['week'].astype(float)
data['weight'] = data['weight'].astype(float)
data['rbg'] = data['rbg'].astype(float)

# Define custm joint model function
def joint_model(train_long, train_surv, test_long, test_surv):
    # Create a mixed effects model for the longitudinal data
    with pm.Model() as joint_model:
        # Priors for fixed effects
        beta_weight = pm.Normal('beta_weight', mu=0, sigma=10)
        beta_week = pm.Normal('beta_week', mu=0, sigma=10)
        beta_sex = pm.Normal('beta_sex', mu=0, sigma=10)
        beta_diet = pm.Normal('beta_diet', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Random effects
        unique_ids = train_long['NR_Name'].cat.categories
        n_ids = len(unique_ids)
        u_intercept = pm.Normal('u_intercept', mu=0, sigma=1, shape=n_ids)
        u_weight = pm.Normal('u_weight', mu=0, sigma=1, shape=n_ids)
        u_week = pm.Normal('u_week', mu=0, sigma=1, shape=n_ids)
        u_sex = pm.Normal('u_sex', mu=0, sigma=1, shape=n_ids)
        u_diet = pm.Normal('u_diet', mu=0, sigma=1, shape=n_ids)
        id_idx = train_long['NR_Name'].cat.codes.values

        # Expected value
        mu = (beta_weight * train_long['weight'].values +
              beta_week * train_long['week'].values +
              beta_sex * train_long['sex'].values +
              beta_diet * train_long['diet'].values +
              u_intercept[id_idx] +
              u_weight[id_idx] * train_long['weight'].values +
              u_week[id_idx] * train_long['week'].values +
              u_sex[id_idx] * train_long['sex'].values +
              u_diet[id_idx] * train_long['diet'].values)
        
        # Likelihood
        rbg_obs = pm.Normal('rbg_obs', mu=mu, sigma=sigma, observed=train_long['rbg'].values)
        trace = pm.sample(1000, tune=1000, cores=1, return_inferencedata=False)

    # Fit Cox Proportional Hazards model for survival data
    cph = CoxPHFitter()
    cph.fit(train_surv, duration_col='Event_time', event_col='Event', formula="sex + diet")
    return trace, cph

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
for i, (train_index, test_index) in enumerate(kf.split(data['NR_Name'].unique())):
    print(f"Processing fold {i+1}")
    train_ids = data['NR_Name'].unique()[train_index]
    test_ids = data['NR_Name'].unique()[test_index]

    train_long = data[data['NR_Name'].isin(train_ids)]
    test_long = data[data['NR_Name'].isin(test_ids)]
    train_surv = train_long[['NR_Name', 'Event_time', 'Event', 'sex', 'diet']].drop_duplicates()
    test_surv = test_long[['NR_Name', 'Event_time', 'Event', 'sex', 'diet']].drop_duplicates()
    train_overall = train_long.groupby('NR_Name').agg({'rbg': 'mean', 'weight': 'mean', 'week': 'mean'}).reset_index()
    test_overall = test_long.groupby('NR_Name').agg({'rbg': 'mean', 'weight': 'mean', 'week': 'mean'}).reset_index()
    train_overall = pd.merge(train_overall, train_surv, on='NR_Name')
    test_overall = pd.merge(test_overall, test_surv, on='NR_Name')
    trace, cox_fit = joint_model(train_long, train_surv, test_long, test_surv)

    # Predictions for longitudinal data
    test_long = test_long.copy()
    test_long['pred'] = (np.mean(trace['beta_weight']) * test_long['weight'] +
                         np.mean(trace['beta_week']) * test_long['week'] +
                         np.mean(trace['beta_sex']) * test_long['sex'] +
                         np.mean(trace['beta_diet']) * test_long['diet'])
    preds_long = test_long.groupby('NR_Name').agg({'pred': 'mean'}).reset_index()
    preds_long = pd.merge(preds_long, test_overall[['NR_Name', 'rbg']], on='NR_Name')
    rmse = np.sqrt(mean_squared_error(preds_long['rbg'], preds_long['pred']))
    mae = mean_absolute_error(preds_long['rbg'], preds_long['pred'])

    # C-index calculation using lifelines
    preds_surv = test_surv.copy()
    preds_surv['pred'] = cox_fit.predict_partial_hazard(test_surv)
    cindex = cox_fit.concordance_index_
    print(f"Fold {i+1} - RMSE: {rmse}, MAE: {mae}, C-index: {cindex}")
    results.append({'Fold': i+1, 'RMSE': rmse, 'MAE': mae, 'Cindex': cindex})

results_df = pd.DataFrame(results)
print(results_df)
# Note: The above code is a simplified version and may require further adjustments based on the 
# specific dataset and requirements.
