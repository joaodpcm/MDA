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
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from sklearn.inspection import permutation_importance
import requests
import numpy as np
import itertools as it
from plotly.subplots import make_subplots
import base64
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_hist_gradient_boosting


# #importing models
# url_class = 'https://api.github.com/repos/joaodpcm/MDA/contents/classifier.pkl'
# response_class = requests.get(url_class)
# data_class = response_class.json()
# content_class = data_class['content']
# decoded_content_class = base64.b64decode(content_class)

# rfc = pickle.loads(decoded_content_class)

# url_reg = 'https://api.github.com/repos/joaodpcm/MDA/contents/regressor.pkl'
# response_reg = requests.get(url_reg)
# data_reg = response_reg.json()
# content_reg = data_reg['content']
# decoded_content_reg = base64.b64decode(content_reg)

# hgr = pickle.loads(decoded_content_reg)


# Importing models locally
rfc = pickle.load(open('classifier.pkl', 'rb'))
hgr = pickle.load(open('regressor.pkl', 'rb'))




#importing avarage of the noise

url_hourly_avg = 'https://raw.githubusercontent.com/joaodpcm/MDA/master/avg_hourly_noise.csv'

df_hourly_avg=pd.read_csv(url_hourly_avg)

# Retrieve the current date and time
current_time = datetime.now()

# # Create a range of dates and times for the next 48 hours
# time_range = pd.date_range(start=current_time, periods=48, freq='H')
# # Filter the dataset for the next 48 hours
# filtered_data = df_hourly_avg[
#     (df_hourly_avg['DayOfWeek'] == current_time.weekday()) &
#     (df_hourly_avg['HourOfDay'].isin(time_range.hour))
# ]


# Create a range of dates and times for the next 48 hours
time_range = [current_time + timedelta(hours=x) for x in range(48)]
time_range_df = pd.DataFrame()
time_range_df['time'] = [i.replace(second=0, microsecond=0, minute=0, hour=i.hour) for i in time_range]
time_range_df['avg'] = np.zeros(48)
for i in range(len(time_range)):
    time_range_df.loc[i, 'avg'] = df_hourly_avg[
    (df_hourly_avg['DayOfWeek'] == time_range_df.loc[i, 'time'].dayofweek) &
    (df_hourly_avg['HourOfDay'] == time_range_df.loc[i, 'time'].hour)
].AverageNoise.values[0]

#getting data for amount of events
events_url = 'https://raw.githubusercontent.com/joaodpcm/MDA/master/shaped_filter_tags_city2_EXAM.csv'
events=pd.read_csv(events_url,sep='\t')
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
forecast['events_count'] = events_48['Events']

#making prediction on unseen data
prediction_reg = hgr.predict(forecast)

prediction_class1 = rfc.predict(forecast)
prediction_class = [x + 1 for x in prediction_class1]


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

#making a graph for the classifier
start_time = datetime.now()
hours_list = [start_time + timedelta(hours=x) for x in range(48)]

colors = {1: "green", 2: "yellow", 3: "red"}
category_order = ["Low", "Intermediate", "High"]
fig_class = go.Figure(data=[go.Bar(x=hours_list,
                            y=prediction_class,
#                             marker=dict(color='red'),
                            marker=dict(color=[colors[level] for level in prediction_class]),
                            customdata=prediction_class,
                            hovertemplate="Hour: %{x}<br>Noise Level: %{customdata}<extra></extra>")])
fig_class.update_xaxes(title_text="Hours")
fig_class.update_yaxes(title_text="Noise Level", categoryorder="array", categoryarray=category_order)
fig_class.update_layout(title_text="Relative Noise Levels in the Next 48 Hours")


#making a graph for the regressor

# create coordinate  pairs
x_pairs = pairwise(time_range)
y_pairs = pairwise(prediction_reg)

# generate color list
colors_list=[colors[level] for level in prediction_class]

fig_reg = go.Figure()

# add traces (line segments)
for x, y, color in zip(x_pairs, y_pairs, colors_list):
    fig_reg.add_trace(
        go.Scatter(
            x=x,
            y=y, 
            mode='lines', 
            line={'color': color, 'width':7},
        showlegend=False
        )
    )
fig_reg.add_trace(go.Scatter(x=time_range, y=time_range_df['avg'], mode='lines', name='Average Noise',
    line=dict(dash='dash', color='grey', width=3)))

fig_reg.update_xaxes(title_text="Hours")
fig_reg.update_yaxes(title_text="dB")
fig_reg.update_layout(title_text="Noise Forecast vs Avarage for the next 48 hours")

trace1 = go.Scatter(x=time_range, y=prediction_reg, name='Noise Forecast', line={'color':'blue'})
trace2 = go.Scatter(x=time_range, y=time_range_df['avg'], mode='lines', name='Average Noise',
    line=dict(dash='dash', color='grey', width=3))
trace3 = go.Bar(x=hours_list,
                            y=[65]*48,
                            # marker=dict(color='red'),
                            marker=dict(color=colors_list, opacity=0.2),
                            customdata=prediction_class,
                            hovertemplate="Hour: %{x}<br>Noise Level: %{customdata}<extra></extra>",
                            showlegend=False)
fig_reg_2 = go.Figure(data = [trace1, trace2, trace3])


fig_reg_2.layout.bargap = 0.
fig_reg_2.update_yaxes(showticklabels=True)
fig_reg_2.update_layout(yaxis_range=[38,65])

fig_reg_2.update_xaxes(title_text="Hours")
fig_reg_2.update_yaxes(title_text="dB")
fig_reg_2.update_layout(title_text="Noise Forecast vs Avarage for the next 48 hours")




#App

with st.sidebar.container():
    st.title("Netherlands Team")

    with st.expander('About the project'):
        st.header("objective")

    with st.expander('Authors'):
        st.write("")



st.title('Noise forecast')

# Create the child tabs within the parent tab
with st.expander('Events on the next days?'):
    # Create a table with checkboxes for each hour
    st.header("Are there any events on the next two days?")
    selected_hours = st.multiselect('Select hours for the event', time_range_df['time'], default=[])
    # Update the 'Event' column based on the selected hours
    forecast.loc[forecast['hour'].isin(selected_hours), 'Event'] = True
    #Update tag
    for hour in selected_hours:
        event_type = st.selectbox(f'Select event type for {hour}',['Party', 'Sports', 'Cultural', 'Pub Crawl'])
        forecast.loc[forecast['hour']==hour, 'tag_category']=event_type

with st.expander('Bar plot'):
    st.write('This is the content of Child Tab 1')
    st.header("Noise levels for the next 2 days")
    st.plotly_chart(fig_class)
    st.markdown(""" This graph shows a categorical prediction for the noise level for the next 48 hours relative to the usual noise levels on these hours. 
        '<span style="color:red">The red bars indicate hours that will be louder than usual.</span>'
        '<span style="color:yellow"> The yellow bars indicate hours that will be like the usual.</span>'
         '<span style="color:green"> The green bars indicate hours that will be calmer than usual.</span>' """,unsafe_allow_html=True)


with st.expander('Noise level with regression model'):
    st.header('Noise level with regression model')
    st.plotly_chart(fig_reg)
    st.markdown('This graph shows the absolute levels of noise expected for the next 48 hours in the continuous line, and the avarage of these hours in the dotted line. The colors in the line show the classification of the classifier model on that hour')


with st.expander('Noise level with regression model and classifier on the background'):
    st.header('Noise level with regression model and classifier on the background')
    st.plotly_chart(fig_reg_2)
    st.markdown('This shows a comparison between the absolute values and the relative prediction')



