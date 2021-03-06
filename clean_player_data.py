import pandas as pd
import numpy as np
import requests

def clean(player_data_df):
  '''
    Helper function that cleans the scraped player data.

          Parameters:
                player_data_df (pandas.DataFrame): scraped player data to clean
              
          Returns:
                player_data_df (pandas.DataFrame): cleaned player data
              
  '''

  player_data_df['HEIGHT_IN'] = np.where(player_data_df.HEIGHT != np.nan, (
      (player_data_df.HEIGHT.str.split('-').str[0].astype(float) * 12) + 
      (player_data_df.HEIGHT.str.split('-').str[-1].astype(float)) 
      ),
      player_data_df.HEIGHT
  )

  player_data_df['LAST_AFFILIATION1'] = np.where(player_data_df.LAST_AFFILIATION.str.contains('/'),
                                              player_data_df.LAST_AFFILIATION.str.split('/').str[0],
                                              player_data_df.LAST_AFFILIATION)

  player_data_df['LAST_AFFILIATION2'] = np.where(player_data_df.LAST_AFFILIATION.str.contains('/'),
                                              player_data_df.LAST_AFFILIATION.str.split('/').str[-1],
                                              np.nan)

  player_data_df['PRIMARY_POSITION'] = np.where(player_data_df.POSITION_INITIALS.str.contains('-'),
                                                player_data_df.POSITION_INITIALS.str.split('-').str[0],
                                                player_data_df.POSITION_INITIALS)
  
  player_data_df.BIRTHDATE = pd.to_datetime(player_data_df.BIRTHDATE)

  player_data_df.WEIGHT = np.where(player_data_df.WEIGHT == '-', np.nan, player_data_df.WEIGHT)

  player_data_df.HEADSHOT_URL = [
                             i 
                             if requests.get(i).status_code == 200 
                             else 'https://stats.wnba.com/media/img/league/wnba-logo-fallback.png' 
                             for i in player_data_df.HEADSHOT_URL
                             ]

  player_data_df = player_data_df.fillna(value=np.nan)

  player_data_df.columns = [i.lower() for i in player_data_df.columns]

  player_data_df = (player_data_df
                    .rename(columns={
                        'person_id':'player_id',
                        'birthdate':'birth_date',
                        'display_fi_last':'fi_last',
                        'rosterstatus':'roster_status',
                        'display_first_last':'player_name',
                        })
                    )

  player_data_df = player_data_df[
                                  ['player_id','player_name','first_name','last_name','fi_last',
                                  'birth_date','age','age_days','height','height_in','weight',
                                  'primary_position','position_initials','jersey','school','country',
                                  'season_exp','from_year','to_year','roster_status','team_id','team_city',
                                  'team_name','team_abbreviation','draft_year','draft_round','draft_number',
                                  'last_affiliation','last_affiliation1','last_affiliation2','headshot_url'
                                  ]
                                ]

  return player_data_df
