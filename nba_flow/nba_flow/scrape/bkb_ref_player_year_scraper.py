import urllib2 as ul
from bs4 import BeautifulSoup
import pandas as pd

url = "http://www.basketball-reference.com/leagues/NBA_2013_advanced.html"

# this is the html from the given url
html = ul.urlopen(url)
soup = BeautifulSoup(html)
column_headers = [th.getText() for th in 
                  soup.findAll('tr')[0].findAll('th')[1:]]

url_template = "http://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
player_df = pd.DataFrame()

for year in range(1995, 2016):  # for each year
    url = url_template.format(year=year)  # get the url
    
    html = ul.urlopen(url)  # get the html
    soup = BeautifulSoup(html) # create our BS object
    
    # get our player data
    data_rows = soup.findAll('tr')[1:] 
    player_data = [[td.getText() for td in data_rows[i].findAll('td')]
                for i in range(len(data_rows))]
    
    # Turn yearly data into a DatFrame
    year_df = pd.DataFrame(player_data, columns=column_headers)
    # create and insert the Draft_Yr column
    year_df.insert(0, 'Draft_Yr', year)
    
    # Append to the big dataframe
    player_df = player_df.append(year_df, ignore_index=True)

player_df = player_df.convert_objects(convert_numeric=True)

# Get rid of the rows full of null values
player_df = player_df[player_df.Player.notnull()]

# Replace NaNs with 0s
player_df = player_df.fillna(0)

# Drop null columns
cols = [19,24]
player_df.drop(player_df.columns[cols],axis=1,inplace=True)

# Rename Columns
player_df.rename(columns={'WS/48':'WS_per_48'}, inplace=True)
# Change % symbol
player_df.columns = player_df.columns.str.replace('%', '_Perc')

# Changing the Data Types to int
player_df.loc[:,'G':'MP'] = player_df.loc[:,'G':'MP'].astype(int)
player_df.loc[:,'Age':'Age'] = player_df.loc[:,'Age':'Age'].astype(int)

player_df.to_csv("player_data_1995_to_2015.csv", index=False)
