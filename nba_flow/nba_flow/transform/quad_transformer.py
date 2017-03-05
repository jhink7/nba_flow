import pandas as pd
import numpy as np

df_raw = pd.read_csv('raw_data/player_data_1995_to_2015.csv')

# Select only the columns we care about
df_ws = df_raw[['Year','Player', 'Age', 'Tm', 'MP', 'WS_per_48']]

# for split seasons, only keep the total
df_ws = df_ws.groupby(['Player', 'Year']).first().reset_index()

# construct a dataframe player vs year (mps and ws)
piv_ws = df_ws.pivot(index = 'Player', columns='Year', values='WS_per_48').reset_index()
piv_mp = df_ws.pivot(index = 'Player', columns='Year', values='MP').reset_index()
piv_age = df_ws.pivot(index = 'Player', columns='Year', values='Age').reset_index()
df_by_year = pd.merge(piv_ws, piv_mp, how='left', on=['Player'], suffixes=('_ws', '_mp')).merge(piv_age, how='left', on=['Player'], suffixes=('_age', '_age'))

#rename age columns

df_by_year.columns.values[43:64] = [str(df_by_year.columns.values[43:64][col]) + 
                                  "_age" for col in range(21)]

# rearrange columns by alpha order
df_by_year = df_by_year.sort_index(axis=1)


# construct all 4 valid 4 year player sets (year1_stats + year2_stats + year3_stats + age  + target_year_stats)
# this is the data that we'll build our model from
df_all_valid = pd.DataFrame()
for year in range(1998, 2016):
    df_curr_year = df_by_year[['Player', 
                               str(year) + '_age',
                               str(year-3) + '_ws', 
                               str(year-3) + '_mp', 
                               str(year-2) + '_ws', 
                               str(year-2) + '_mp', 
                               str(year-1) + '_ws', 
                               str(year-1) + '_mp', 
                               str(year) + '_ws', 
                               str(year) + '_mp']]
    #df_curr_year.to_csv("temp.csv", index=False)
    df_curr_year.columns = ['player', 'age', 'ws_3', 'mp_3', 'ws_2', 'mp_2', 'ws_1', 'mp_1', 'ws_target', 'mp_target']
    df_curr_year = df_curr_year[df_curr_year.ws_3.notnull() & df_curr_year.ws_2.notnull() & df_curr_year.ws_1.notnull() & df_curr_year.ws_target.notnull()]
    df_all_valid = df_all_valid.append(df_curr_year, ignore_index=True)

df_all_valid.to_csv("ws_quads_all.csv", index=False)
