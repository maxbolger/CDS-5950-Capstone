import pandas as pd
import numpy as np

import time
import requests
from bs4 import BeautifulSoup as bs

def main(file_type='csv'):
  '''
    Scrapes player position, height, and weight for players that did not have
    this information listed on the WNBA players page and updates the existing
    player data
    
          Parameters:
                file_type (str): file type for data to be saved as
  '''
  players = pd.read_parquet('wnba_player_data.parquet')
  df = pd.read_parquet('wnba_shot_data.parquet')

  players_list_name = players.loc[players.primary_position.isna()==True].player_name.to_list()
  players_list_id = players.loc[players.primary_position.isna()==True].player_id.to_list()

  player_data = []
  urls = [
          'b/barksla01w' if
          i == 'LaQuanda Quick' else
          'm/mccarst01w' if
          i == 'Stephanie White' else
          'w/washito01w' if
          i == 'Tonya Massaline' else
          (i.split()[-1][0] + '/' + i.replace('.','').split()[-1][:5] + i.replace('.','').split()[0][:2] + '01w').lower() for
          i in players_list_name
          ]

  for player, id, url in zip(players_list_name, players_list_id, urls):
    print(f'Scraping {player}...')
    r = requests.get(f'https://www.basketball-reference.com/wnba/players/{url}.html')
    soup = bs(r.content)
    # data = soup.find_all("p")

    player_data.append(
        {
            'player_name': player,
            'player_id': id,
            'position_initials': [
                                i for 
                                i in 
                                soup.find_all("p") if 
                                'Position:' in 
                                ' '.join(i.get_text(separator=' ').split())
                                ][0].get_text(separator=' ').split()[-1],
            'height': soup.find("span", itemprop="height").get_text() if len(soup.find_all("span", itemprop="height")) > 0 else np.nan,
            'weight': soup.find("span", itemprop="weight").get_text() if len(soup.find_all("span", itemprop="weight")) > 0 else np.nan
        }
    )

    time.sleep(5)

  print('Done!')

  player_data_df = pd.DataFrame.from_dict(player_data, orient='columns')

  player_data_df.position_initials = (
      player_data_df.position_initials
      .str.replace('Guard','G')
      .str.replace('Forward','F')
      .str.replace('Center','C')
      )
  
  player_data_df['primary_position'] = (np
                                        .where(
                                            player_data_df.position_initials.str.contains('-'),
                                            player_data_df.position_initials.str.split('-').str[0],
                                            player_data_df.position_initials)
                                        )

  player_data_df['height_in'] = (np
                                .where(
                                    player_data_df.height != np.nan,
                                    (
                                      (player_data_df.height.str.split('-').str[0].astype(float) * 12) + 
                                      (player_data_df.height.str.split('-').str[-1].astype(float))
                                      ),
                                      player_data_df.height
                                      )
                                )
  
  player_data_df['weight'] = (np
                                .where(
                                    player_data_df.weight != np.nan,
                                    player_data_df.weight.str.replace('lb','').astype(float),
                                    player_data_df.weight
                                    )
                                )

  player_data_df = player_data_df.set_index('player_id')
  players = players.set_index('player_id')
  players.update(player_data_df)
  players = players.reset_index()

  players.at[9, 'player_name'] = 'Michelle Campbell (100360)'
  players.at[722, 'player_name'] =  'Michelle Campbell (202314)'

  players.at[31,'headshot_url'] = 'https://ak-static.cms.nba.com/wp-content/uploads/headshots/wnba/100797.png'

  # 202270 Angel Robinson never attempted a shot in the WNBA amd was released after being drafted.
  # 202657 Angel Robinson played for SEA in 2014 then PHO in 2017-2018
  # The WNBA api credits 202270 for 202657's shots during her SEA stint
  # As a result, for the purpose of this data, I am overwriting 202270's player data with 202657's
  players = players.drop(index=846)

  angel = {
      'age': 34,'age_days': 351,'birth_date': '1987-04-20','country': 'USA','draft_number': '20','draft_round': '2',
      'draft_year': '2010','fi_last': 'A. Robinson','first_name': 'Angel','from_year': 2014.0,
      'headshot_url': 'https://stats.wnba.com/media/img/league/wnba-logo-fallback.png','height': '6-6','height_in': 78.0,
      'jersey': 0.0,'last_affiliation': 'Georgia/USA','last_affiliation1': 'Georgia','last_affiliation2': 'USA',
      'last_name': 'Robinson','player_id': 202270,'player_name': 'Angel Robinson','position_initials': 'F-C',
      'primary_position': 'F','roster_status': 'Inactive','school': 'Georgia','season_exp': 4,'team_abbreviation': 'PHO',
      'team_city': 'Phoenix','team_id': 1611661317,'team_name': 'Mercury','to_year': 2018.0,'weight': 193.0
      }

  players = players.append(angel, ignore_index=True)

  if file_type == 'csv':
    print(f'Now saving the data as a {file_type} file...')
    players.to_csv('wnba_player_data_updated.csv',index=False)
  elif file_type == 'parquet':
    print(f'Now saving the data as a {file_type} file...')
    players.to_parquet('wnba_player_data_updated.parquet',index=False)
  else:
    print('No valid file type given. Defaulting to csv')
    players.to_csv('wnba_player_data_updated.csv',index=False)

if __name__ == "__main__":
  main(file_type='csv')
