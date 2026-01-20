import sys
import subprocess
import pandas as pd
from io import StringIO
from copy import deepcopy

def list_tables(accdb_file): # List all tables in the Access database
    result = subprocess.run(['mdb-tables', '-1', accdb_file], capture_output=True, text=True)
    return result.stdout.strip().split('\n')

def export_table_as_csv(accdb_file, table_name): # Export a specific table from the Access database as CSV
    result = subprocess.run(['mdb-export', accdb_file, table_name], capture_output=True, text=True)
    return result.stdout

# Read the Access database and convert tables to pandas DataFrames
access_db_file = '20250622_NileRat.accdb'

tables = list_tables(access_db_file) # List all tables in the Access database

csv_data = export_table_as_csv(access_db_file, 'main') # Export the main table
main = pd.read_csv(StringIO(csv_data)).convert_dtypes() # Convert to pandas Dataframe

main['birth_date'] = pd.to_datetime(main['birth_date'], format='%m/%d/%y %H:%M:%S').dt.date # Convert birth_date to datetime.date
main['death_date'] = pd.to_datetime(main['death_date'], format='%m/%d/%y %H:%M:%S').dt.date # Same for death_date

# Reorder columns to match the original order
main = main[["NR_ID","NR_Name","sex","generation","father","mother","foster","gestational_diet","nursing_diet","weanling_diet","majority_diet","overall_diet","birth_date","death_date","litter_size","litter_order","notes"]]

csv_data = export_table_as_csv(access_db_file, 'measurement') # Export the measurement table
measurement = pd.read_csv(StringIO(csv_data)).convert_dtypes() # Convert to pandas Dataframe
measurement = measurement[measurement["week"] % 4 == 0]
measurement['expected_date'] = pd.to_datetime(measurement['expected_date'], format='%m/%d/%y %H:%M:%S').dt.date
measurement['actual_date'] = pd.to_datetime(measurement['actual_date'], format='%m/%d/%y %H:%M:%S').dt.date
measurement['measurement_notes'] = measurement['measurement_notes'].astype('str')
measurement['next_diet'] = measurement['next_diet'].astype('str')

dtypeFix = {'weight': 'float64', 'rbg': 'float64', 'weight percentile': 'float64', 'pregnant_days': 'Int64'}
for i in range(len(measurement.columns)):
    if measurement.columns[i] in dtypeFix:
        measurement[measurement.columns[i]] = pd.to_numeric(
            measurement[measurement.columns[i]], errors='coerce').round().astype(
                dtypeFix[measurement.columns[i]])

measurement = measurement[["measurement_ID","NR_Name","week","expected_date","actual_date","weight","rbg","weight percentile","diet","next_diet","pregnant_days","measurement_notes","Plasma","LN2","RNAlater"]]

# Select rows where rbg > 200 from measurements table
grabRBG = deepcopy(measurement[['NR_Name', 'week', 'rbg']])

rbg200 = grabRBG[grabRBG['rbg'] > 200]
rbg200_min = rbg200.loc[rbg200.groupby('NR_Name')['week'].idxmin()].rename(columns={'week': 'rbg200'})

# Do the same for 100, 120, and 300
rbg100 = grabRBG[grabRBG['rbg'] > 100]
rbg100_min = rbg100.loc[rbg100.groupby('NR_Name')['week'].idxmin()].rename(columns={'week': 'rbg100'})
rbg120 = grabRBG[grabRBG['rbg'] > 120]
rbg120_min = rbg120.loc[rbg120.groupby('NR_Name')['week'].idxmin()].rename(columns={'week': 'rbg120'})
rbg300 = grabRBG[grabRBG['rbg'] > 300]
rbg300_min = rbg300.loc[rbg300.groupby('NR_Name')['week'].idxmin()].rename(columns={'week': 'rbg300'})

# Merge the weeks with the main table
onset = deepcopy(main.merge(rbg100_min[["NR_Name", "rbg100"]], on='NR_Name', how='left'))
onset = onset.merge(rbg120_min[["NR_Name", "rbg120"]], on='NR_Name', how='left')
onset = onset.merge(rbg200_min[["NR_Name", "rbg200"]], on='NR_Name', how='left')
onset = onset.merge(rbg300_min[["NR_Name", "rbg300"]], on='NR_Name', how='left')

# Fill the NaN values with 80
onset['rbg100'] = onset['rbg100'].fillna(80).astype('Int64')
onset['rbg120'] = onset['rbg120'].fillna(80).astype('Int64')
onset['rbg200'] = onset['rbg200'].fillna(80).astype('Int64')
onset['rbg300'] = onset['rbg300'].fillna(80).astype('Int64')

csv_data = export_table_as_csv(access_db_file, 'breeding')
breeding = pd.read_csv(StringIO(csv_data)).convert_dtypes()

breeding['notes'] = breeding['notes'].astype('str')
dtypeFix = {'d0_count': 'Int64', 'd7_count': 'Int64', 'd14_count': 'Int64', 'd21_count': 'Int64',
            'd0_weight': 'float64', 'd7_weight': 'float64', 'd14_weight': 'float64', 'd21_weight': 'float64',
            'maternal_onset': 'Int64', 'paternal_onset': 'Int64'}
for i in range(len(breeding.columns)):
    if breeding.columns[i] in dtypeFix:
        breeding[breeding.columns[i]] = pd.to_numeric(
            breeding[breeding.columns[i]], errors='coerce').round().astype(dtypeFix[breeding.columns[i]])

dDateFix = ['mating_date', 'pups_born', 'weaned_date', 'd0_date', 'd7_date', 'd14_date', 'd21_date']
for i in range(len(breeding.columns)):
    if breeding.columns[i] in dDateFix:
        breeding[breeding.columns[i]] = pd.to_datetime(
            breeding[breeding.columns[i]], format='%m/%d/%y %H:%M:%S').dt.date

babyReadings = breeding.iloc[:, 10:22].dropna(how = 'all')

if len(sys.argv) > 1:
    main.to_csv('main.csv', index = False)
    measurement.to_csv('measurement.csv', index = False)
    onset.to_csv('onset.csv', index = False)