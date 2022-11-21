
import pandas as pd
import numpy as np
import timeit
import os
import json
from joblib import load
from scipy.sparse import data
import subprocess
import sys
import prepocessing as pp

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 
##################Function to get model predictions
def model_predictions(dataset_path):
    #read the deployed model and a test dataset, calculate predictions
    model = load(os.path.join(model_path, "trainedmodel.pkl"))
    encoder = load(os.path.join(model_path, "encoder.pkl"))
    
    if dataset_path is None: dataset_path = "testdata.csv"
    df = pd.read_csv(os.path.join(test_data_path, dataset_path))

    df_x, df_y, _ = pp.preprocess_data(df, encoder)

    y_pred = model.predict(df_x)

    return y_pred, df_y

##################Function to get summary statistics
def dataframe_summary():
     #calculate summary statistics here
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    numeric_columns = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees"
        ]
    
    res = []
    for column in numeric_columns:
        res.append([column, "mean", df[column].mean()])
        res.append([column, "median", df[column].median()])
        res.append([column, "standard deviation", df[column].std()])
    
    return res

#################### Missing Data

def missing_data():
    #calculate summary statistics here
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    
    result = []
    for column in df.columns:
        count_na = df[column].isna().sum()
        count_not_na = df[column].count()
        count_total = count_not_na + count_na

        result.append([column, str(int(count_na/count_total*100))+"%"])
    
    return str(result)

##################Function to get timings
def execution_time():
    res = []
    for procedure in ["training.py" , "ingestion.py"]:
        starttime = timeit.default_timer()
        os.system('python3 %s' % procedure)
        timing=timeit.default_timer() - starttime
        res.append([procedure, timing])
 
    return str(res)

##################Function to check dependencies
def outdated_packages_list():
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    
    return str(outdated_packages)


if __name__ == '__main__':
    model_predictions(None)
    
    execution_time()
    
    dataframe_summary()
    
    missing_data()
    
    outdated_packages_list()





    
