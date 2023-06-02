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
import scipy.interpolate as sp
import re
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

#loading unseen data
content = "https://weather.com/weather/tenday/l/8c6b0e55d5cf8568f60d839eaf3fa128975a8daf414f334c76ea19e9e1e1d3b0"
response = requests.get(content)
soup = BeautifulSoup(response.content, 'html.parser')

temp_val = soup.findAll('div', class_ ='DailyForecast--DisclosureList--nosQS')
next13 = temp_val[0].text.split('|')[-26:]
temp = [float(re.findall(r'\d+', next13[i].split('.')[0])[0]) for i in range(0, len(next13))]
humid = [float(re.findall(r'\d+', next13[i].split('.')[3])[0]) for i in range(0, len(next13))]

hour, hours = np.arange(0, 26*12, 12), np.arange(0, 26*12, 1)
t = sp.interp1d(hour, temp, kind='linear', fill_value = 'extrapolate')
h = sp.interp1d(hour, humid, kind='linear', fill_value = 'extrapolate')
temperature, humidity = t(hours), h(hours)

# Create a range of dates and times for the next two weeks
first_day = current_time + timedelta(days=2)
year = first_day.year
month = first_day.month
day = first_day.day
first_hour = pd.to_datetime(f'{year}-{month}-{day} 12:00:00')
time_range = [first_hour + timedelta(hours=int(x)) for x in hours]
time_range_df = pd.DataFrame()
time_range_df['time'] = [i.replace(second=0, microsecond=0, minute=0, hour=i.hour) for i in time_range]
time_range_df['avg'] = np.zeros(len(hours))
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

# Load future events
events_location = '/home/dave/Documents/uni/modern_data/project/MDA/data_fetched_filter_tags_city_EXAM.csv'
df_events_full = pd.read_csv(events_location, sep='\t')
df_events_full['startTime'] = pd.to_datetime(df_events_full['startTime'])
df_events_full['startTime'] = [i.replace(second=0, microsecond=0, minute=0, hour=i.hour) for i in df_events_full['startTime']]
df_events_full.loc[0, 'startTime'] = pd.to_datetime('2023-06-09 07:00:00')
df_events_full.loc[3, 'startTime'] = pd.to_datetime('2023-06-15 07:00:00')
df_events_full_48 =  df_events_full[df_events_full['startTime'].isin([i for i in time_range_df['time']])].reset_index()

median = 10   # in mm
def rain_converter(perc, median):
    if perc < 30:
        return 0
    else:
        return perc/100*1.3*median

forecast = pd.DataFrame()
forecast['temp'] = temperature
forecast['humidity'] = humidity
forecast['wind'] = np.concatenate([float(re.findall(r'\d+', next13[i].split('.')[0])[2])*np.ones(12) for i in range(0, len(next13))])
forecast['rain'] = np.concatenate([float(re.findall(r'\d+', next13[i].split('.')[0])[1])*np.ones(12) for i in range(0, len(next13))])
forecast['rain'] = [rain_converter(i, median) for i in forecast['rain']]
weekday = [(first_hour+timedelta(hours=int(i))).weekday() for i in hours]
hour_of_day = [(first_hour+timedelta(hours=int(i))).hour for i in hours]
forecast['nameday'] = weekday
forecast['hour'] = hour_of_day
forecast['time'] = time_range
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
hours_list = [start_time + timedelta(hours=x) for x in range(len(hours))]

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
fig_class.update_layout(title_text="Relative Noise Levels in the Next two weeks")


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
fig_reg.update_layout(title_text="Noise Forecast vs Avarage for the next two weeks")

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


# Weather plot
weather_fig = go.Figure()



time_weather = [first_hour + timedelta(hours=int(x)) for x in hour]

weather_fig.add_trace(go.Scatter(x=time_weather, y=temp, mode='lines', name='Temperature',
    line=dict(color='red', width=2), yaxis='y1', line_shape='spline'))
weather_fig.add_trace(go.Scatter(x=time_weather[1:-1], y=humid[1:-1], mode='lines', name='Humidity',
    line=dict(color='green', width=2), yaxis='y2', line_shape='spline'))
weather_fig.add_trace(go.Bar(x=time_range, y=forecast['rain'], name='rain [mm]', text=forecast['rain'], textposition='outside',
                             marker_color='blue', opacity=0.4,  marker_line_color='blue', marker_line_width=5, yaxis='y3'))


weather_fig.update_layout(
    xaxis=dict(
        domain=[0, 0.8]
    ),
    yaxis=dict(
        title="<b>Temperature [°C]</b>",
        titlefont=dict(
            color="red"
        ),
        tickfont=dict(
            color="red"
        )
    ),
    
    yaxis2=dict(
        title="<b>Humidity [%]</b>",
        titlefont=dict(
            color="green"
        ),
        tickfont=dict(
            color="green"
        ),
        overlaying="y", # specifyinfg y - axis has to be separated
        side="right", # specifying the side the axis should be present
    ),

    yaxis3=dict(
        title="<b>Rain amount [mm]</b>",
        titlefont=dict(
            color="blue"
        ),
        tickfont=dict(
            color="blue"
        ),
        anchor="free",  # specifying x - axis has to be the fixed
        overlaying="y", # specifyinfg y - axis has to be separated
        side="right", # specifying the side the axis should be present
        range=[0, 50],
        position=0.93
        # visible=False
    ),

)




#App

st.markdown("""
<style>
.big-font {
    font-size:25px !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar.container():
    st.title("Netherlands Team")

    with st.expander('About the project'):
        st.header("objective")

    with st.expander('Authors'):
        st.write("")



st.title('Noise forecast from ' + time_range[0].strftime("%B %d") + ' to ' + time_range[-1].strftime("%B %d"))

with st.expander('Bar plot'):
    st.write('This is the content of Child Tab 1')
    st.header("Noise levels for the next 2 days")
    st.plotly_chart(fig_class)
    st.markdown(""" This graph shows a categorical prediction for the noise level for the next 48 hours relative to the usual noise levels on these hours. 
        '<span style="color:red">The red bars indicate hours that will be louder than usual.</span>'
        '<span style="color:yellow"> The yellow bars indicate hours that will be like the usual.</span>'
         '<span style="color:green"> The green bars indicate hours that will be calmer than usual.</span>' """,unsafe_allow_html=True)

st.header('Noise level with regression model')
st.plotly_chart(fig_reg)
st.markdown('This graph shows the absolute levels of noise expected for the next 48 hours in the continuous line, and the avarage of these hours in the dotted line. The colors in the line show the classification of the classifier model on that hour')



st.markdown('<p class="big-font">We have found the following events in the next 2 weeks:</p>', unsafe_allow_html=True)
for i in range(len(df_events_full_48)):
    st.write('- ', df_events_full_48.loc[i, 'title'], ', on ', 
        df_events_full_48.loc[i, 'startTime'].strftime("%B"), str(df_events_full_48.loc[i, 'startTime'].day), 
        ' at ', str(df_events_full_48.loc[i, 'startTime'].strftime("%I:%M %p")),
        ' (link: ', df_events_full_48.loc[i, 'url'], ')')



# st.write('We have found the following events in the next 48 hours:')
# Create the child tabs within the parent tab
with st.expander('Do you know about any other event?'):
    # Create a table with checkboxes for each hour
    # st.header("Are there any events on the next two days?")
    selected_hours = st.multiselect('Select hours for the event', time_range_df['time'].dt.strftime("%B %d,  %I:%M %p"), default=[])
    # Update the 'Event' column based on the selected hours
    forecast.loc[forecast['time'].isin(selected_hours), 'Event'] = True
    #Update tag
    for hour in selected_hours:
        event_type = st.selectbox(f'Select event type for {hour}',['Party', 'Sports', 'Cultural', 'Pub Crawl'])
        forecast.loc[forecast['time']==hour, 'tag_category']=event_type



with st.expander('Noise level with regression model and classifier on the background'):
    st.header('Noise level with regression model and classifier on the background')
    st.plotly_chart(fig_reg_2)
    st.markdown('This shows a comparison between the absolute values and the relative prediction')

st.markdown('<p class="big-font">The weather will be like:</p>', unsafe_allow_html=True)
st.plotly_chart(weather_fig)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: 100%;
        background-position: relative;
        background-opacity: 0.01;
        background-color: rgba(0,0,0,0.25);
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('background_lowOp_blur.jpg')  



