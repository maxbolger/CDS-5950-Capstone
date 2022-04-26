import pandas as pd
import numpy as np
import requests
import logging
import time
import os

import matplotlib.pyplot as plt

import category_encoders as ce
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,log_loss,precision_score,recall_score,f1_score,roc_auc_score

from lightgbm import LGBMClassifier

from preprocess_data import preprocess
from logger import log

def main(file_type='csv'):
  """
  This file preprocesses the shot data and creates the final model. 
  A logfile of scores is saved along with the data with predictions.
  """
  logger = log(path=os.getcwd(), file="model.log")

  data = pd.read_parquet('wnba_shot_data.parquet')

  df = preprocess(data)

  X = df[
         ['shot_distance','action_type','loc_x','loc_y','era_1','era_2','era_3','era_4',
          'era_5','era_6','home','gp','time','buzzer','bubble','primary_position']
         ]

  y = df['shot_made_flag']

  pipe = make_pipeline(ce.OrdinalEncoder(), LGBMClassifier())

  start = time.process_time()
  pipe.fit(X, y)
  end = time.process_time()

  preds = pipe.predict_proba(X)

  display = CalibrationDisplay.from_estimator(
        pipe,
        X,
        y,
        n_bins=10,
    )
  plt.close()

  acc = round(accuracy_score(y,np.round(preds[:,1])),4)
  ll = round(log_loss(y,preds[:,1]),4)
  p = round(precision_score(y,np.round(preds[:,1])),4)
  r = round(recall_score(y,np.round(preds[:,1])),4)
  f1 = round(f1_score(y,np.round(preds[:,1])),4)
  ra = round(roc_auc_score(y,np.round(preds[:,1])),4)
  cal = round(np.absolute((display.prob_pred - display.prob_true).mean()),4)
  imps = list(dict(zip(pipe.steps[1][1].feature_importances_,X.columns)).items())
  t = round(end-start,2)

  metrics = {
      'Accuracy Score':acc,
      'Log Loss':ll,
      'Precision Score':p,
      'Recall Score':r,
      'F1 Score':f1,
      'Roc Auc Score':ra,
      'Calibration Score':cal,
      'Feature Importances':imps,
      'Fit Time':t
  }

  logger.info("Model Results:")

  for i,j in metrics.items():
    logger.info(f"{i}: {j}")
    logger.info('--------------------------')

  preds_df = pd.DataFrame(pipe.predict_proba(X),index=X.index,columns=['pred_miss','pred_make']).iloc[:,-1]

  df = df.join(preds_df)

  if file_type == 'csv':
    print(f'Now saving the data as a {file_type} file...')
    df.to_csv('wnba_shot_model_preds.csv',index=False)
  elif file_type == 'parquet':
    print(f'Now saving the data as a {file_type} file...')
    df.to_parquet('wnba_shot_model_preds.parquet',index=False)
  else:
    print('No valid file type given. Defaulting to csv')
    df.to_csv('wnba_shot_model_preds.csv',index=False)

if __name__ == "__main__":
  main(file_type='parquet')
