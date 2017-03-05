import pandas as pd
import numpy as np

df_raw = pd.read_csv('raw_data/player_data_1995_to_2015.csv')

#print df_raw.head()

# Select only the columns we care about
df_ws = df_raw[['Year','Player', 'Age', 'Tm', 'MP', 'WS_per_48']]

# for split seasons, only keep the total
df_ws = df_ws.groupby(['Player', 'Year']).first().reset_index()

#df_ws = df_ws[df_ws.Player == 'Vin Baker']

piv_ws = df_ws.pivot(index = 'Player', columns='Year', values='WS_per_48').reset_index()
piv_mp = df_ws.pivot(index = 'Player', columns='Year', values='MP').reset_index()
piv_age = df_ws.pivot(index = 'Player', columns='Year', values='Age').reset_index()
result = pd.merge(piv_ws, piv_mp, how='left', on=['Player'], suffixes=('_ws', '_mp')).merge(piv_age, how='left', on=['Player'], suffixes=('_age', '_age'))

#rename age columns

result.columns.values[43:64] = [str(result.columns.values[43:64][col]) + 
                                  "_age" for col in range(21)]

# rearrange columns by alpha order
result = result.sort_index(axis=1)

result.to_csv("ws_wide.csv", index=False)

