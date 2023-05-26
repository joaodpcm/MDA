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
import pickle
import requests
from datetime import datetime
from datetime import timedelta
import sklearn
import plotly.graph_objects as go

#importing models
with open('classifier_trained_model.pkl', 'rb') as f:
    rfc = pickle.load(f)

with open('regressor_trained_model.pkl', 'rb') as f:
    hgr = pickle.load(f)


#loading unseen data
content = "https://weather.com/weather/hourbyhour/l/c097b546627cdff2da1e276cb9b2731055718a5e7270d777a92857a9701c7870"
response = requests.get(content)
soup = BeautifulSoup(response.content, 'html.parser')

temp_val = soup.findAll('div', attrs={'class':'DetailsTable--field--CPpc_'})


forecast = pd.DataFrame()
forecast['temp'] = [round((int(temp_val[i].text[-3:-1])-32 ) *5/9, 1) for i in list(np.array(range(288))) if i%6 == 0]
forecast['wind'] = [float(temp_val[i].text.split(' ')[1]) for i in list(np.array(range(288))) if i%6 == 1]
forecast['wind_direction'] = [temp_val[i].text.split(' ')[0] for i in list(np.array(range(288))) if i%6 == 1]
forecast['humidity'] = [int(temp_val[i].text[-3:-1]) for i in list(np.array(range(288))) if i%6 == 2]
forecast['cloud_cover'] = [int(temp_val[i].text.replace('Cloud Cover', '')[:-1]) for i in list(np.array(range(288))) if i%6 == 4]
forecast['rain'] = [int(temp_val[i].text.replace('Rain Amount', '').replace(' in', '')) for i in list(np.array(range(288))) if i%6 == 5]
weekday = [(datetime.now()+timedelta(hours=i)).weekday() for i in range(48)]
hour_of_day = [(datetime.now()+timedelta(hours=i)).hour for i in range(48)]
forecast['nameday'] = weekday
forecast['hour'] = hour_of_day
forecast['event_yes'] = False # This value has to be included by the user. So edit this. The value now is missing, but the model running, so even if nothing is provided, it will run
forecast['tag_category'] = 'No event' # This value has to be included by the user. So edit this



#making prediction on unseen data
prediction_reg = hgr.predict(forecast)

prediction_class = rfc.predict(forecast)

#making a graph for the classifier
colors = {"Low": "green", "Intermediate": "yellow", "High": "red"}
category_order = ["Low", "Intermediate", "High"]
fig_class = go.Figure(data=[go.Bar(x=forecast['hour'],
                            y=prediction_class,
                            marker=dict(color=[colors[level] for level in prediction_class]),
                            customdata=prediction_class,
                            hovertemplate="Hour: %{x}<br>Noise Level: %{customdata}<extra></extra>")])
fig_class.update_xaxes(title_text="Hours")
fig_class.update_yaxes(title_text="Noise Level", categoryorder="array", categoryarray=category_order)
fig_class.update_layout(title_text="Noise Levels in the Next 48 Hours")


#making a graph for the regressor
fig_reg = go.Figure(data=go.Scatter(x=forecast['hour'], y=prediction_reg))

fig_reg.update_xaxes(title_text="Hours")
fig_reg.update_yaxes(title_text="dB")
fig_reg.update_layout(title_text="Noise levels in the Next 48 Hours")

#App
st.title("Noise forecast")

# Create a table with checkboxes for each hour
hours = list(range(48))
st.header("Are there any events on the next two days?")
selected_hours = st.multiselect('Select hours for the event', hours, default=[])
# Update the 'Event' column based on the selected hours
forecast.loc[forecast['hour'].isin(selected_hours), 'Event'] = True
#Update tag
for hour in selected_hours:
    event_type = st.selectbox(f'Select event type for {hour}',['Party', 'Sports', 'Cultural', 'Pub Crawl'])
    forecast.loc[forecast['hour']==hour, 'tag_category']=event_type

st.header("Noise levels for the next 2 days")
st.plotly_chart(fig_class)
st.plotly_chart(fig_reg)
