import pandas as pd
import numpy as np

def main(file_type='csv'):
  """
  Prepares model output data for streamlit app
  """

  df = pd.read_parquet('wnba_shot_model_preds.parquet')

  df.player_id = (np
                    .where(df.player_name == 'Angel Robinson',
                          202657,
                          df.player_id)
                  )
  df.player_id = (np
                    .where(df.player_name == 'Claudia Neves',
                          100386,
                          df.player_id)
                  )

  TYPES = {'2PT Field Goal':2,'3PT Field Goal':3}

  df['value'] = df.shot_type.map(TYPES)

  df['fgoe'] = ((df.shot_made_flag - df['pred_make']) * 100).astype(float).round(2)
  df['outcome'] = df.shot_made_flag * df['value']
  df['xpts'] = df['value'] * df['pred_make']
  df['poe'] = df['outcome'] - df['xpts']

  df = df[
          ['game_id','team_id','player_id','player_name','gp','loc_x','loc_y','shot_zone_basic',
           'shot_zone_range','action_type','shot_made_flag','year','value','outcome','pred_make']
          ]

  if file_type == 'csv':
    print(f'Now saving the data as a {file_type} file...')
    df.to_csv('app_data.csv',index=False)
  elif file_type == 'parquet':
    print(f'Now saving the data as a {file_type} file...')
    df.to_parquet('app_data.parquet',index=False)
  else:
    print('No valid file type given. Defaulting to csv')
    df.to_csv('app_data.csv',index=False)

if __name__ == "__main__":
  main(file_type='parquet')
