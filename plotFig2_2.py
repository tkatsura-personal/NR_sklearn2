
from plotFig2_1 import pivot_M, pivot_F, weeks, onset_sex_rab, onset_sex_rod
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from copy import deepcopy

# Plot survival curves for same data using line plots
# Remove 80 from weeks for better x-axis representation
weeks = [w for w in weeks if w != 80]
pivot_M_reverse = deepcopy(pivot_M.drop(index=80))
pivot_F_reverse = deepcopy(pivot_F.drop(index=80))

# Subtract from 100 to get cumulative percentage
pivot_M_reverse = 100 - pivot_M_reverse
pivot_F_reverse = 100 - pivot_F_reverse

# Calculate hazard ratio using Cox Proportional Hazards model
from lifelines import CoxPHFitter
# Prepare data for Cox model
# Combine onset_sex_rab and onset_sex_rod
onset_sex_rod = deepcopy(onset_sex_rod)
onset_sex_rab = deepcopy(onset_sex_rab)
onset_sex_rod['diet'] = 'Rod'
onset_sex_rab['diet'] = 'Rab'
onset_sex_combined = pd.concat([onset_sex_rod, onset_sex_rab], ignore_index=True)

# Separate into male and female
onset_sex_M = onset_sex_combined[onset_sex_combined['sex'] == 'M']
onset_sex_F = onset_sex_combined[onset_sex_combined['sex'] == 'F']

# Create duration and event columns (duration = rbg200 weeks, event = 1 if rbg200 <= 40 else 0, 
# then replace rbg200 = 80 with 40))
onset_sex_M['event'] = np.where(onset_sex_M['rbg200'] <= 40, 1, 0)
onset_sex_M['duration'] = onset_sex_M['rbg200'].replace(80, 40)

# Convert sex into numerical values for Cox model
onset_sex_M['diet'] = onset_sex_M['diet'].map({'Rab': 1, 'Rod': 0})
# Fit Cox model
cph_M = CoxPHFitter()
cph_M.fit(onset_sex_M[['duration', 'event', 'diet']], duration_col='duration', event_col='event', show_progress=True)
cph_M.print_summary()

# Do the same for female
onset_sex_F['event'] = np.where(onset_sex_F['rbg200'] <= 40, 1, 0)
onset_sex_F['duration'] = onset_sex_F['rbg200'].replace(80, 40)
onset_sex_F['diet'] = onset_sex_F['diet'].map({'Rab': 1, 'Rod': 0})
cph_F = CoxPHFitter()
cph_F.fit(onset_sex_F[['duration', 'event', 'diet']], duration_col='duration', event_col='event', show_progress=True)
cph_F.print_summary()

# Write the onset_sex_M and onset_sex_F dataframes to csv for later use
onset_sex_M.to_csv('CoxPH_M.csv', index=False)
onset_sex_F.to_csv('CoxPH_F.csv', index=False)

# Change font globally for better aesthetics
mpl.rcParams['font.family'] = 'Arial'

# Plot survival curves (Ratio 1:1 for each plot, total width = 12, height = 10)
fig, ax = plt.subplots(2, 1, sharex = True, height_ratios = [1, 1], figsize=(12, 10))
# Male
# Plot lines with markers and different colors for Rab and Rod
ax[0].plot(weeks, pivot_M_reverse.Rab.to_list(), marker='o', color=(120/256, 166/256, 107/256), label='High-fiber maternal diet')
ax[0].plot(weeks, pivot_M_reverse.Rod.to_list(), marker='o', color=(255/256, 166/256, 28/256), label='Regular chow maternal diet')

# Set labels, title, legend, grid, and axis limits
ax[0].set_ylabel('Diabetes-free Survival (%)', fontsize=22, fontweight='bold') # Change font size of y-axis label and bold them
ax[0].legend(reverse = True) # Reverse the order of legend to match the order of lines
ax[0].grid(axis='y') # Add gridlines only to y-axis for better readability
ax[0].set_title('Male offspring', fontsize=30, x = 0.2, y = 0.88) # Change font size of title and adjust y position
ax[0].set_xlim(4, 44) # Set x-axis limits
ax[0].set_xticks(np.arange(8, 44, 4)) # Set x-axis ticks
ax[0].set_ylim(0, 120) # Set y-axis limits
ax[0].set_yticks(np.arange(0, 125, 25)) # Set y-axis ticks
ax[0].tick_params(axis='both', labelsize=18) # Change font size of tick labels
plt.setp(ax[0].get_legend().get_texts(), fontsize='16') # Change font size of legend text

# Female
ax[1].plot(weeks, pivot_F_reverse.Rab.to_list(), marker='o', color=(120/256, 166/256, 107/256), label='High-fiber maternal diet')
ax[1].plot(weeks, pivot_F_reverse.Rod.to_list(), marker='o', color=(255/256, 166/256, 28/256), label='Regular chow maternal diet')

# Set labels, title, legend, grid, and axis limits
ax[1].set_xlabel('Age (weeks)', fontsize=22, fontweight='bold') # Change x-axis label to "Week"
ax[1].set_ylabel('Diabetes-free Survival (%)', fontsize=22, fontweight='bold') # Change font size of y-axis label
ax[1].legend(reverse = True) # Reverse the order of legend to match the order of lines
ax[1].grid(axis='y') # Add gridlines only to y-axis for better readability
ax[1].set_title('Female offspring', fontsize=30, x = 0.2, y = 0.88) # Change font size of title and adjust y position
ax[1].set_xlim(4, 44) # Set x-axis limits
ax[1].set_xticks(np.arange(8, 44, 4)) # Set x-axis ticks
ax[1].set_ylim(0, 120) # Set y-axis limits
ax[1].set_yticks(np.arange(0, 125, 25)) # Set y-axis ticks
ax[1].tick_params(axis='both', labelsize=18) # Change font size of tick labels
plt.setp(ax[1].get_legend().get_texts(), fontsize='16') # Change font size of legend text

# Draw 95% confidence intervals as shaded areas
MRab = np.array(pivot_M_reverse.Rab.to_list())
MRod = np.array(pivot_M_reverse.Rod.to_list())
FRab = np.array(pivot_F_reverse.Rab.to_list())
FRod = np.array(pivot_F_reverse.Rod.to_list())
# Male 
ax[0].fill_between(weeks, 
                  MRab - 1.96 * (MRab * (100 - MRab) / len(onset_sex_M))**0.5,
                  MRab + 1.96 * (MRab * (100 - MRab) / len(onset_sex_M))**0.5,
                  color=(120/256, 166/256, 107/256), alpha=0.2)
ax[0].fill_between(weeks, 
                  MRod - 1.96 * (MRod * (100 - MRod) / len(onset_sex_M))**0.5,
                  MRod + 1.96 * (MRod * (100 - MRod) / len(onset_sex_M))**0.5,
                  color=(255/256, 166/256, 28/256), alpha=0.2)
# Female
ax[1].fill_between(weeks, 
                  FRab - 1.96 * (FRab * (100 - FRab) / len(onset_sex_F))**0.5,
                  FRab + 1.96 * (FRab * (100 - FRab) / len(onset_sex_F))**0.5,
                  color=(120/256, 166/256, 107/256), alpha=0.2)
ax[1].fill_between(weeks, 
                  FRod - 1.96 * (FRod * (100 - FRod) / len(onset_sex_F))**0.5,
                  FRod + 1.96 * (FRod * (100 - FRod) / len(onset_sex_F))**0.5,
                  color=(255/256, 166/256, 28/256), alpha=0.2)

plt.tight_layout()
plt.show()

