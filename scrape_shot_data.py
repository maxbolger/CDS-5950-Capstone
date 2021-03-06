import pandas as pd
import numpy as np
import requests
import json
import time

def main(file_type='csv'):
  '''
    Scrapes WNBA shot data for the years specificed by the user.
    Data is saved as a csv file to the users current directory.

          Parameters:
              file_type (str): file type for data to be saved as
              (default, 'csv')
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
                         ' AppleWebKit/537.36 (KHTML, like Gecko)' +\
                         ' Chrome/81.0.4044.129 Safari/537.36'}

  try:
    start = int(input('Which year would you like to start collecting data (Min. 1997): '))
    while start < 1997:
      start = int(input('Year must be greater than or equal to 1997: '))
  except ValueError:
    start = int(input('Input must be a number. Which year would you like to start collecting data (Min. 1997): '))
  
  try:
    stop = int(input('Which year would you like to end collecting data (Max. 2021): '))
    while stop > 2021:
      stop = int(input('Year must be less than or equal to 2021: '))
  except ValueError:
    stop = int(input('Input must be a number. Which year would you like to end collecting data (Max. 2021): '))
    
  print(f'Scraping data from {start}-{stop}')

  YEARS = reversed(np.arange(start,stop+1,1))
  df = pd.DataFrame()

  for i in YEARS:
    print(f'Scraping {i}...')

    url = "https://stats.wnba.com/stats/shotchartdetail?Period=0&VsConference&LeagueID=10&LastNGames=0&TeamID=0" \
          "&PlayerPosition&Location&Outcome&ContextMeasure=FGA&DateFrom&StartPeriod&DateTo&OpponentTeamID=0" \
          f"&ContextFilter&RangeType&Season={i}&AheadBehind&PlayerID=0&EndRange&VsDivision&PointDiff&RookieYear" \
          "&GameSegment&Month=0&ClutchTime&EndPeriod&SeasonType=Regular%20Season&SeasonSegment&GameID"

    r = requests.get(url,headers=HEADERS)
    js = json.loads(r.text)

    df_i = pd.DataFrame(js['resultSets'][0]['rowSet'],columns=js['resultSets'][0]['headers'])
    df_i['year'] = i

    df = df.append(df_i)
    print(f'Done scraping {i}.')
    time.sleep(3)

  df.columns = [i.lower() for i in df.columns]

  if file_type == 'csv':
    print(f'Now saving the data as a {file_type} file...')
    df.to_csv('wnba_shot_data.csv',index=False)
  elif file_type == 'parquet':
    print(f'Now saving the data as a {file_type} file...')
    df.to_parquet('wnba_shot_data.parquet',index=False)
  else:
    print('No valid file type given. Data has not been downloaded to the current directory')

  print('Complete!')


if __name__ == "__main__":
  main(file_type='parquet')
