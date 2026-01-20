import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyread import measurement, onset
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Grab where week = 4 and rbg > 200
measurement = measurement[measurement['week'] == 8]
measurement = measurement[measurement['rbg'] > 200]

#write to csv
#measurement.to_csv('measurement_week8_rbg200.csv', index=False)
onset = onset[onset['rbg200'] == 8]


nursing_count = onset[onset['gestational_diet'] == "Rod"].shape[0]
print(nursing_count)

#write onset to csv
#onset.to_csv('onset_week8_rbg200.csv', index=False)