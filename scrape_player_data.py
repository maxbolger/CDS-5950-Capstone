import sys
import json
import time
import pickle
import random
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs

from clean_player_data import clean

def main(file_type='csv',player_ids='wnba_player_ids.pkl'):
  '''
    Scrapes WNBA player data given a list of player ids.
    A list of dictionaries is pickled and dumped to users current directory.
    A cleaned csv is also saved to the users current directory.
    
            Parameters:
                file_type (str): file type for data to be saved as
                player_ids (*.pkl): player ids pkl file           
  '''

  HEADERS = {'Connection': 'keep-alive',
            'Host': 'stats.wnba.com',
            'Origin': 'http://stats.wnba.com',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'stats.wnba.com',
            'x-nba-stats-origin': 'stats',
            'x-nba-stats-token': 'true',
            'Accept-Language': 'en-US,en;q=0.9',
            "X-NewRelic-ID": "VQECWF5UChAHUlNTBwgBVw==",
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6)' +\
                          ' AppleWebKit/537.36 (KHTML, like Gecko)' + \
                          ' Chrome/81.0.4044.129 Safari/537.36'}
  try:
    with open(player_ids, 'rb') as f:
      PLAYER_IDS = pickle.load(f)
  except FileNotFoundError:
    raise SystemExit(
        "It looks like you don't have the player_ids pickle file in your cwd. " \
        "Make sure it is there and try again."
        )

  player_data = []

  for index, player_id in enumerate(PLAYER_IDS):
    r = requests.get(f'https://stats.wnba.com/player/{player_id}/',headers=HEADERS)
    soup = bs(r.content)

    script_tags = soup.findAll("script")

    if index == 1:
      for script_tag in script_tags:
        if "window.wteams" in script_tag.text:
          windows = script_tag.text.split(" = ")
          team_info_start = windows[1]
          team_info_replaced = team_info_start.replace(';\n  ', "")
          team_info_json = json.loads(team_info_replaced)
          with open('wnba_team_data_raw.pkl', 'wb') as f:
            pickle.dump(team_info_json, f)
          break
    else:
      pass

    for script_tag in script_tags:
      if "window.nbaStatsPlayerInfo" in script_tag.text:
        windows = script_tag.text.split(" = ")
        player_info_start = windows[1]
        player_info_clean = player_info_start.replace(';\n  window.nbaStatsPlayerStats', "")
        player_info_json = json.loads(player_info_clean)
        player_info_json.update(
            {"HEADSHOT_URL":f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/wnba/{player_id}.png"}
            )   
        player_data.append(player_info_json)
        break

    if index % 50 == 0:
      print(f'{index} of {len(PLAYER_IDS)} iterations completed...')

    time.sleep(random.uniform(0, 1))

  with open('wnba_player_data_raw.pkl', 'wb') as f:
    pickle.dump(player_data, f)

  print('Cleaning data...')
  player_data_df = pd.DataFrame.from_dict(player_data, orient='columns').replace(r'^\s*$', np.nan, regex=True)
  player_data_df = clean(player_data_df)

  if file_type == 'csv':
    print(f'Now saving the data as a {file_type} file...')
    player_data_df.to_csv('wnba_player_data.csv',index=False)
  elif file_type == 'parquet':
    print(f'Now saving the data as a {file_type} file...')
    player_data_df.to_parquet('wnba_player_data.parquet',index=False)
  else:
    print('No valid file type given. Data has not been downloaded to the current directory')

  print('Done!')

if __name__ == "__main__":
  main(file_type='parquet')
