#importing packages
import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from selenium import webdriver
from bs4 import BeautifulSoup
import numpy as np 
import seaborn as sns
import plotly.graph_objects as go
import missingno as msno
import matplotlib.pyplot as plt
import os
import sklearn

#weather web scraping
driver = webdriver.Chrome("~/Downloads/chromedriver")
driver.get("https://weather.com/weather/hourbyhour/l/c097b546627cdff2da1e276cb9b2731055718a5e7270d777a92857a9701c7870")
content = driver.page_source
soup = BeautifulSoup(content)
temp_val = soup.findAll('span', attrs={'class':'DetailsSummary--tempValue--jEiXE'})
hour = soup.findAll('h3', attrs={'class':'DetailsSummary--daypartName--kbngc'})
wind = soup.findAll('span', attrs={'class':'Wind--windWrapper--3Ly7c DetailsTable--value--2YD0-'})


print( temp_val[0].text[:-1] , hour[0].text , wind[0].text)



#loading datasets and merging
df_meteo = pd.read_csv('meteo_data/data_final_meteo.csv')
df_noise = pd.read_csv('noise_data/noise_data_complete')
df_noise = df_noise.rename(columns={"result_timestamp": "DATEUTC", "comp": "target"})
df = df_meteo.merge(df_noise, how='inner', on='DATEUTC')
df['DATEUTC'] = pd.to_datetime(df['DATEUTC'])
df['nameday'] = df['DATEUTC'].dt.dayofweek


# Splitting Data
# Extracting correct features
from sklearn.model_selection import train_test_split
x = df[['LC_TEMP_QCL3', 'LC_HUMIDITY', 'LC_WINDSPEED', 'LC_RAININ',
       'LC_DAILYRAIN', 'nameday']]
  
y = df['target']

# Splitting data into train data and validation data 
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)

#Random Forest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from sklearn.inspection import permutation_importance


# Define numeric and categorical features
numeric_features = ['LC_TEMP_QCL3', 'LC_HUMIDITY', 'LC_WINDSPEED', 'LC_RAININ',
       'LC_DAILYRAIN',]
categorical_features = ['nameday']

# Define transformers for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create preprocessor for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create pipeline with preprocessor and random forest regressor
rfc = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestClassifier(n_estimators=300, random_state=42))
])

# Fit the pipeline on the training data
rfc.fit(X_train, y_train)

# Make predictions on the test data
y_pred_rfc = rfc.predict(X_test)

# Fit the OneHotEncoder transformer to the categorical features
categorical_transformer.fit(X_train[categorical_features])

# Get feature importances and names
importances = rfc.named_steps['regressor'].feature_importances_
encoded_cat_features = categorical_transformer.named_steps['onehot'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(encoded_cat_features)
feature_importances = np.zeros(len(all_feature_names))

# Add the importances of the original numeric features
feature_importances[:len(numeric_features)] += importances[:len(numeric_features)]

# Combine the importances of the encoded categorical features into the original features
feature_importances = importances[:len(numeric_features)].tolist()  # Start with the numeric features
for i, feature_name in enumerate(categorical_features):
    encoded_cat_importances = [
        importances[j] for j, feat_name in enumerate(all_feature_names)
        if feat_name.startswith(feature_name + '_')
    ]
    feature_importances.append(sum(encoded_cat_importances))

# Get the names of the original features
original_feature_names = numeric_features + categorical_features


# The online data is named unseen
unseen = [temp_val[0].text[:-1], hour[0].text, wind[0].text]

# Generate predictions on the test set
prediction_app = rfc.predict(unseen)
print(temp_val[0].text[:-1], hour[0].text, wind[0].text)
print(prediction_app)


'''
#App
st.title("Noise forecast")

st.write("The next two days will be:")
st.write(prediction_app)

with st.sidebar:
    st.header('Calendar')
    st.header('Weather')

d = st.date_input(
    "Select a date",
    datetime.date(2019, 7, 6))


st.header('Weather')
st.dataframe(weather)
'''