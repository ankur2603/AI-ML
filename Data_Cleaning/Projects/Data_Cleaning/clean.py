import numpy as np
import pandas as pd


df = pd.read_excel("unclean_data.xlsx")

df.drop(columns=['title_year.1'], inplace=True)
df.drop(columns=['facenumber_in_poster'], inplace=True)

df['DIRECTOR_facebook_likes'] = pd.to_numeric(df['DIRECTOR_facebook_likes'],errors='coerce')

df['duration'].fillna(df['duration'].median(), inplace=True)
df['DIRECTOR_facebook_likes'].fillna(df['DIRECTOR_facebook_likes'].median(),inplace=True)
df['num_voted_users'].fillna(df['num_voted_users'].median(), inplace=True)
df['ACTOR_2_facebook_likes'].fillna(df['ACTOR_2_facebook_likes'].median(), inplace=True)
df['Cast_Total_facebook_likes'].fillna(df['Cast_Total_facebook_likes'].median(), inplace=True)

df.to_excel("cleaned_data.xlsx", index=False)

print(df.isnull().sum(), "\n")
print(df.info())
