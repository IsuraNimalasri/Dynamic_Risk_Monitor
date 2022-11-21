from flask import Flask, session, jsonify, request,render_template
import pandas as pd
import numpy as np
import diagnostics 
import json
import os
from scoring import score_model


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None

@app.route("/",methods=['GET'])
def home():
    return render_template("index.html")

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    datapath = request.json.get('dataset_path')
    y_pred, _ = diagnostics.model_predictions(datapath)
    return str(y_pred)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed mode
    score = score_model()
    return str(score)
#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary():        
    #check means, medians, and modes for each col
    summary = diagnostics.dataframe_summary()
    return str(summary)
#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats():        
    #check timing and percent NA values
    
    et = diagnostics.execution_time()
    md = diagnostics.missing_data()
    op = diagnostics.outdated_packages_list()     
    return str("execution_time:" + et + "\nmissing_data;"+ md + "\noutdated_packages:" + op)
if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
