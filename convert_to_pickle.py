import pandas as pd

filename = 'New York Citibike Trips.zip'

df_raw = pd.read_csv(filename, header=0)
df = df_raw.copy(deep=True)
df.to_pickle('New York Citibike Trips.pkl')    #to save the dataframe, df to .pkl file