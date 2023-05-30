#importing packages
import streamlit as st
import pandas as pd
import datetime
from selenium import webdriver
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from datetime import timedelta
import requests
import numpy as np
import base64

#importing models
url_class = 'https://api.github.com/repos/joaodpcm/MDA/contents/classifier.pkl'
response_class = requests.get(url_class)
data_class = response_class.json()
content_class = data_class['content']
decoded_content_class = base64.b64decode(content_class)

rfc = pickle.loads(decoded_content_class)

url_reg = 'https://api.github.com/repos/joaodpcm/MDA/contents/regressor.pkl'
response_reg = requests.get(url_reg)
data_reg = response_reg.json()
content_reg = data_reg['content']
decoded_content_reg = base64.b64decode(content_reg)

hgr = pickle.loads(decoded_content_reg)



# url_class = 'https://raw.githubusercontent.com/joaodpcm/MDA/master/classifier.pkl'
# response_class = requests.get(url_class)
# rfc = pickle.load(open('classifier.pkl','rb'))

# url_reg= 'https://raw.githubusercontent.com/joaodpcm/MDA/master/regressor.pkl'
# response_reg= requests.get(url_reg)
# hgr = pickle.load(open('regressor.pkl','rb'))

# with open('classifier.pkl', 'wb') as f:
#     f.write(response_class.content)

# with open('classifier.pkl', 'rb') as f:
#     rfc = pickle.load(f)

# with open('regressor.pkl','wb') as f:
#     f.write(response_reg.content)

# with open('regressor.pkl', 'rb') as f:
#     hgr = pickle.load(f)


#importing avarage of the noise
df_hourly_avg=pd.read_csv('noise_data/hourly_acg_noise.csv')

# Retrieve the current date and time
current_time = datetime.now()

# Create a range of dates and times for the next 48 hours
time_range = pd.date_range(start=current_time, periods=48, freq='H')
# Filter the dataset for the next 48 hours
filtered_data = df_hourly_avg[
    (df_hourly_avg['DayOfWeek'] == current_time.weekday()) &
    (df_hourly_avg['HourOfDay'].isin(time_range.hour))
]

events=pd.read_csv('shaped_filter_tags_city2_EXAM.csv')
events['Time'] = pd.to_datetime(events['Time'])
events_48 = events[events['Time'].isin(time_range)]



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
forecast['# of events'] = events_48['Events']



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
fig_class.update_layout(title_text="Relative Noise Levels in the Next 48 Hours")


#making a graph for the regressor
fig_reg = go.Figure()
fig_reg.add_trace(data=go.Scatter(x=forecast['hour'], y=prediction_reg, name='Noise Forecast'))
fig_reg.add_trace(go.Scatter(x=time_range, y=filtered_data['AverageNoise'], mode='lines', name='Average Noise',line=dict(dash='dash')))

fig_reg.update_xaxes(title_text="Hours")
fig_reg.update_yaxes(title_text="dB")
fig_reg.update_layout(title_text="Noise Forecast vs Avarage for the next 48 hours")




#App
st.title("Noise forecast")

# Create a table with checkboxes for each hour
st.header("Are there any events on the next two days?")
selected_hours = st.multiselect('Select hours for the event', forecast['hour'], default=[])
# Update the 'Event' column based on the selected hours
forecast.loc[forecast['hour'].isin(selected_hours), 'Event'] = True
#Update tag
for hour in selected_hours:
    event_type = st.selectbox(f'Select event type for {hour}',['Party', 'Sports', 'Cultural', 'Pub Crawl'])
    forecast.loc[forecast['hour']==hour, 'tag_category']=event_type

st.header("Noise levels for the next 2 days")
st.plotly_chart(fig_class)
st.markdown("This graph shows a categorical prediction for the noise level for the next 48 hours relative to the usual noise levels on these hours. "
            '<span style="color:red">The red bars indicate hours that will be louder than usual.</span>'  
            '<span style="color:yellow">The yellow bars indicate hours that will be like the usual.</span>'
             '<span style="color:green">The green bars indicate hours that will be calmer than usual.</span>',unsafe_allow_html=True)
st.plotly_chart(fig_reg)
st.markdown('This graph shows the absolute levels of noise expected for the next 48 hours in the continuous line, and the avarage of these hours in the dotted line')