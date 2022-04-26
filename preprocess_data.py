import pandas as pd
import numpy as np

def preprocess(data):
  """
  This file preprocesses the shot data for model creation
  
  Parameters:
        data (pandas.DataFrame): a df of the orignal shot data to preprocess  
        
  Returns:
        df (pandas.DataFrame): preprocessed data
  """

  df = data.loc[~(data.action_type=='No Shot')].copy()
  df.year = df.year.astype(int)

  players = pd.read_parquet('wnba_player_data_updated.parquet')
  df = df.merge(players[['player_id','primary_position']],on='player_id')

  TEAMS = {
      'Atlanta Dream':'ATL',
      'Charlotte Sting':'CHA',
      'Chicago Sky':'CHI',
      'Cleveland Rockers':'CLE',
      'Connecticut Sun':'CON',
      'Dallas Wings':'DAL',
      'Detroit Shock':'DET',
      'Houston Comets':'HOU',
      'Indiana Fever':'IND',
      'Las Vegas Aces':'LVA',
      'Los Angeles Sparks':'LAS',
      'Miami Sol':'MIA',
      'Minnesota Lynx':'MIN',
      'New York Liberty':'NYL',
      'Orlando Miracle':'ORL',
      'Phoenix Mercury':'PHO',
      'Portland Fire':'POR',
      'Sacramento Monarchs':'SAC',
      'San Antonio Silver Stars':'SAN',
      'San Antonio Stars':'SAN',
      'Seattle Storm':'SEA',
      'Tulsa Shock':'TUL',
      'Utah Starzz':'UTA',
      'Washington Mystics':'WAS'
  }
  df['home'] = np.where(df.team_name.map(TEAMS) == df.htm,1,0)

  df['era'] = 0
  df.loc[(df.year.between(1997,2000)), ['era']] = 1
  df.loc[(df.year.between(2001,2004)), ['era']] = 2
  df.loc[(df.year.between(2005,2008)), ['era']] = 3
  df.loc[(df.year.between(2009,2012)), ['era']] = 4
  df.loc[(df.year.between(2013,2016)), ['era']] = 5
  df.loc[(df.year>=2017), ['era']] = 6
  df  = pd.get_dummies(df, columns=['era'])

  # df = df.sort_values(by=['player_id','game_date'])
  # df = df.assign(gp=df['game_date']!=df['game_date'].shift())
  # df = df.assign(gp=df.groupby('player_id')['gp'].cumsum()).sort_index()

  df = df.sort_values(by=['player_id','game_date'])
  df['gp'] = (df.game_id) != (df.game_id.shift())
  df['gp'] = df.groupby('player_id').gp.cumsum().sort_index()

  df['p_sec_rem'] = ((df.minutes_remaining * 60) + (df.seconds_remaining))

  # readable logic
  # df['time'] = None
  # def secondsPlayed(df):
  #   if df.year >= 2006:
  #     if df.period == 1:
  #       df['time'] = 600 - df.p_sec_rem
  #     elif (df.period > 1) & (df.period < 5):
  #       df['time'] = ((df.period * 10) * 60) - df.p_sec_rem
  #     elif df.period == 5:
  #       df['time'] = (((4 * 10) * 60) + 300) - df.p_sec_rem
  #     elif df.period == 6:
  #       df['time'] = (((4 * 10) * 60) + 600) - df.p_sec_rem
  #     elif df.period == 7:
  #       df['time'] = (((4 * 10) * 60) + 900) - df.p_sec_rem
  #   elif df.year <= 2005:
  #     if df.period == 1:
  #       df['time'] = 1200 - df.p_sec_rem
  #     elif df.period == 2:
  #       df['time'] = 2400 - df.p_sec_rem
  #     elif df.period == 3:
  #       df['time'] = (((2 * 20) * 60) + 300) - df.p_sec_rem
  #     elif df.period == 4:
  #       df['time'] = (((2 * 20) * 60) + 600) - df.p_sec_rem
  #     elif df.period == 5:
  #       df['time'] = (((2 * 20) * 60) + 900) - df.p_sec_rem
  #     elif df.period == 6:
  #       df['time'] = (((2 * 20) * 60) + 1200) - df.p_sec_rem
  #     elif df.period == 7:
  #       df['time'] = (((2 * 20) * 60) + 1500) - df.p_sec_rem
  #   return df
  # df = df.apply(secondsPlayed,axis=1)

  # vectorized version
  df['time'] = (np
                 .where(df['year']<=2005, 
                        np.where(df['period']==1, 
                                 1200, 1800 + 300*df['period']), 
                        np.where(df['period']<5, 
                                 df['period']*600, 1200 + 300*df['period'])
                        )
                 - df['p_sec_rem']
                 )
  
  # no idea why the code doesn't work correctly for this player, but mannually fixing it here
  df.gp = np.where(
      df.player_id==1627644,
      df.gp+1,
      df.
      gp
      )
  
  df['buzzer'] = (np
                .where(
                    (df.minutes_remaining == 0) &
                    (df.seconds_remaining < 1),
                    1,
                    0
                    )
                )

  df['bubble'] = (np
                  .where(
                      (df.year == 2020),
                      1,
                      0
                      )
                  )
  
  df.action_type = (np
                .where(
                    (df.shot_distance == 0) &
                    (df.shot_type == '3PT Field Goal'),
                    'Layup Shot',
                    df.action_type
                    )
                )

  df.action_type = (np
                  .where(
                      (df.shot_distance > 0) &
                      (df.shot_distance < 19 ) &
                      (df.shot_type == '3PT Field Goal'),
                      'Jump Shot',
                      df.action_type
                      )
                  )

  df.shot_zone_basic = (np
                  .where(
                      (df.shot_distance == 0) &
                      (df.shot_type == '3PT Field Goal'),
                      'Restricted Area',
                      df.shot_zone_basic
                      )
                  )

  df.shot_zone_basic = (np
                  .where(
                      (df.shot_distance > 0) &
                      (df.shot_distance < 19 ) &
                      (df.shot_type == '3PT Field Goal'),
                      'Mid-Range',
                      df.shot_zone_basic
                      )
                  )

  df.shot_type = (np
                  .where(
                      (df.shot_distance < 19) &
                      (df.shot_type == '3PT Field Goal'),
                      '2PT Field Goal',
                      df.shot_type
                      )
                  )
  return df
