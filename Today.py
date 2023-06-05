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
import calendar
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




#App

st.markdown("""
<style>
.big-font {
    font-size:25px !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar.container():
    st.markdown("# NoiSense")

    with st.expander('About the project'):
        st.write('**NoiSense** aims to predict the noise level in Leuven, in particular in the area around Naamsesrtaat, \
            using a machine learning approach. \
            We designed the application to take factors that influence human behavior (and thus, noise on the street) \
            into account, such as weather and events being organized in Leuven. Based on information scraped from the internet, \
            the application gives an approximation of the expected noise level.')

    with st.expander('Authors'):
        st.header('Netherlands team')
        st.write("Wasim Ahmed, Davide Dionese, Joao Duarte, David Frost, Ankur Kalita, Tamás Trombitás ")



st.title('Noise forecast for the next two days')



#importing models


# url_reg = 'https://api.github.com/repos/joaodpcm/MDA/contents/regressor.pkl'
# response_reg = requests.get(url_reg)
# data_reg = response_reg.json()
# content_reg = data_reg['content']
# decoded_content_reg = base64.b64decode(content_reg)

# hgr = pickle.loads(decoded_content_reg)

# # Importing models locally
hgr = pickle.load(open('regressor.pkl', 'rb'))



# @st.cache_resource  # Cache the dataframe initialization
def initialise_forecast(_temp_val, time_range):

    forecast = pd.DataFrame()
    forecast['temp'] = [round((int(_temp_val[i].text[-3:-1])-32 ) *5/9, 1) for i in list(np.array(range(288))) if i%6 == 0]
    forecast['wind'] = [float(_temp_val[i].text.split(' ')[1]) for i in list(np.array(range(288))) if i%6 == 1]
    forecast['wind_direction'] = [_temp_val[i].text.split(' ')[0] for i in list(np.array(range(288))) if i%6 == 1]
    forecast['humidity'] = [int(_temp_val[i].text[-3:-1]) for i in list(np.array(range(288))) if i%6 == 2]
    forecast['cloud_cover'] = [int(_temp_val[i].text.replace('Cloud Cover', '')[:-1]) for i in list(np.array(range(288))) if i%6 == 4]
    forecast['rain'] = [int(_temp_val[i].text.replace('Rain Amount', '').replace(' in', '')) *2.54 for i in list(np.array(range(288))) if i%6 == 5]
    weekday = [(datetime.now()+timedelta(hours=i)).weekday() for i in range(48)]
    hour_of_day = [(datetime.now()+timedelta(hours=i)).hour for i in range(48)]
    forecast['nameday'] = weekday
    forecast['hour'] = hour_of_day
    forecast['time'] = time_range
    forecast['time_nice'] = time_range.dt.strftime("%B %d,  %I:%M %p")
    forecast['event_yes'] = False # This value has to be included by the user. So edit this. The value now is missing, but the model running, so even if nothing is provided, it will run
    forecast['tag_category'] = 'No event' # This value has to be included by the user. So edit this
    forecast['events_count'] = [i for i in events_48['Events']]
    return forecast



#importing avarage of the noise
url_hourly_avg = 'https://raw.githubusercontent.com/joaodpcm/MDA/master/hourly_avg_noise.csv'
df_hourly_avg=pd.read_csv(url_hourly_avg)

# Retrieve the current date and time
current_time = datetime.now()

# Load unseen data
content = "https://weather.com/weather/hourbyhour/l/c097b546627cdff2da1e276cb9b2731055718a5e7270d777a92857a9701c7870"
response = requests.get(content)
soup = BeautifulSoup(response.content, 'html.parser')
temp_val = soup.findAll('div', attrs={'class':'DetailsTable--field--CPpc_'})

# Create a range of dates and times for the next 48 hours
time_range = [current_time + timedelta(hours=x) for x in range(48)]
time_range_df = pd.DataFrame()
time_range_df['time'] = [i.replace(second=0, microsecond=0, minute=0, hour=i.hour) for i in time_range]
time_range_df['avg'] = np.zeros(48)
time_range_df['std'] = np.zeros(48)
for i in range(len(time_range)):
    time_range_df.loc[i, 'avg'] = df_hourly_avg[
    (df_hourly_avg['DayOfWeek'] == time_range_df.loc[i, 'time'].dayofweek) &
    (df_hourly_avg['HourOfDay'] == time_range_df.loc[i, 'time'].hour)
].AverageNoise.values[0]
for i in range(len(time_range)):
    time_range_df.loc[i, 'std'] = df_hourly_avg[
    (df_hourly_avg['DayOfWeek'] == time_range_df.loc[i, 'time'].dayofweek) &
    (df_hourly_avg['HourOfDay'] == time_range_df.loc[i, 'time'].hour)
].Std.values[0]

#getting data for amount of events
events_url = 'https://raw.githubusercontent.com/joaodpcm/MDA/master/shaped_filter_tags_city2_EXAM.csv'
events=pd.read_csv(events_url,sep='\t')
events['Time'] = pd.to_datetime(events['Time'])
events_48 = events[events['Time'].isin(time_range_df['time'])]

# Load future events
events_location = 'https://raw.githubusercontent.com/joaodpcm/MDA/master/data_fetched_filter_tags_city_EXAM.csv'
df_events_full = pd.read_csv(events_location, sep='\t')
df_events_full['startTime'] = pd.to_datetime(df_events_full['startTime'])
df_events_full['startTime'] = [i.replace(second=0, microsecond=0, minute=0, hour=i.hour) for i in df_events_full['startTime']]
df_events_full.loc[2, 'startTime'] = pd.to_datetime('2023-06-03 07:00:00')
df_events_full_48 =  df_events_full[df_events_full['startTime'].isin([i for i in time_range_df['time']])].reset_index()


# Initialize the complete forecast dataframe
forecast = initialise_forecast(temp_val, time_range_df['time'])

# Create box in the app for the user to add more events
st.markdown('<p class="big-font">First, please let us know about any event happening in the next 48 hours. We have found the following events:</p>', unsafe_allow_html=True)
if len(df_events_full_48)==0:
    st.write('- No events were found')
else:
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
    selected_hours = st.multiselect('Select hours for the event:', time_range_df['time'].dt.strftime("%B %d,  %I:%M %p"), default=[])
    # Update the 'Event' column based on the selected hours
    # forecast.loc[forecast['time_nice'].isin(selected_hours), 'event_yes'] = True
    #Update number
    for hour in selected_hours:
        event_number = st.number_input(f'How many events on {hour}?', step=1)
        forecast.loc[forecast['time_nice']==hour, 'events_count'] = forecast.loc[forecast['time_nice']==hour, 'events_count'] + event_number





#making prediction on unseen data
prediction_reg = hgr.predict(forecast)



def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

#making a graph for the classifier
start_time = datetime.now()
hours_list = [start_time + timedelta(hours=x) for x in range(48)]


#making a graph for the regressor

# create coordinate  pairs
x_pairs = pairwise(time_range)
y_pairs = pairwise(prediction_reg)

# generate color list
color_list = list(np.zeros(len(prediction_reg)))
for i in range(len(color_list)):
    if prediction_reg[i] > (time_range_df.loc[i, 'avg'] + time_range_df.loc[i, 'std']):
        color_list[i] = 'red'
    # elif (prediction_reg[i] < (time_range_df.loc[i, 'avg'] + time_range_df.loc[i, 'std'])) & (prediction_reg[i] > (time_range_df.loc[i, 'avg'] - time_range_df.loc[i, 'std'])):
    #     color_list[i] = 'yellow'
    elif prediction_reg[i] < (time_range_df.loc[i, 'avg'] - time_range_df.loc[i, 'std']):
        color_list[i] = 'green'
    else:
        color_list[i] = 'yellow'


fig_reg = go.Figure()

# add traces (line segments)
fig_reg.add_trace(go.Scatter(x=time_range, y=time_range_df['avg'],  mode='lines', name='Average and variability',
    line=dict(dash='dash', color='grey', width=3)))
fig_reg.add_trace(go.Scatter(x=time_range, y=time_range_df['avg']+time_range_df['std'], fill='tozeroy', showlegend=False,
    mode='lines', line=dict(dash='dash', color='grey', width=2)))
fig_reg.add_trace(go.Scatter(x=time_range, y=time_range_df['avg']-time_range_df['std'], fill='tozeroy', fillcolor='white', showlegend=False,
    mode='lines', line=dict(dash='dash', color='grey', width=2)))

for x, y, color in zip(x_pairs, y_pairs, color_list):
    fig_reg.add_trace(
        go.Scatter(
            x=x,
            y=y, 
            mode='lines', 
            line={'color': color, 'width':7},
        showlegend=False
        )
    )

fig_reg.update_xaxes(title_text="Hours")
fig_reg.update_yaxes(title_text="dB")
fig_reg.update_layout(title_text="Noise Forecast vs Avarage for the next 48 hours",
    yaxis_range=[min(prediction_reg)-5,max(prediction_reg)+5])



# Weather plot
weather_fig = go.Figure()

weather_fig.add_trace(go.Scatter(x=time_range, y=forecast['temp'], mode='lines', name='Temperature',
    line=dict(color='red', width=3), yaxis='y1'))
weather_fig.add_trace(go.Scatter(x=time_range, y=forecast['humidity'], mode='lines', name='Humidity',
    line=dict(color='green', width=3), yaxis='y2'))
weather_fig.add_trace(go.Bar(x=time_range, y=forecast['rain'], name='rain [mm]', text=forecast['rain'], textposition='outside',
                             marker_color='blue', opacity=0.4,  marker_line_color='blue', marker_line_width=5, yaxis='y3'))


weather_fig.update_layout(
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
        title="yaxis 3",
        titlefont=dict(
            color="#006400"
        ),
        tickfont=dict(
            color="#006400"
        ),
        anchor="x",  # specifying x - axis has to be the fixed
        overlaying="y", # specifyinfg y - axis has to be separated
        side="right", # specifying the side the axis should be present
        range=[0, 50],
        visible=False
    ),

)



st.header('Do you want to know the noise level at a specific time?')
hours_single = st.selectbox('Select the hour:', time_range_df['time'].dt.strftime("%B %d,  %I:%M %p"))
index_hour = forecast.loc[forecast['time_nice']==hours_single].index.values[0]
st.subheader(f'The noise level will be: {prediction_reg[index_hour]:.1f} dB.')
if prediction_reg[index_hour] > (time_range_df.loc[index_hour, 'avg'] + time_range_df.loc[index_hour, 'std']):
    st.subheader('It will be more noisy than usual.')
# elif (prediction_reg[index_hour] < (time_range_df.loc[index_hour, 'avg'] + time_range_df.loc[index_hour, 'std'])) & (prediction_reg[index_hour] > (time_range_df.loc[index_hour, 'avg'] + 0.4* time_range_df.loc[index_hour, 'std'])):
#     st.subheader('It is going to be slightly louder than usual.')
# elif (prediction_reg[index_hour] < (time_range_df.loc[index_hour, 'avg'] + 0.4 * time_range_df.loc[index_hour, 'std'])) & (prediction_reg[index_hour] > (time_range_df.loc[index_hour, 'avg'] - 0.4* time_range_df.loc[index_hour, 'std'])):
#     st.subheader('It is going to be as noisy as usual')
# elif (prediction_reg[index_hour] < (time_range_df.loc[index_hour, 'avg'] - 0.4 * time_range_df.loc[index_hour, 'std'])) & (prediction_reg[index_hour] > (time_range_df.loc[index_hour, 'avg'] - time_range_df.loc[index_hour, 'std'])):
#     st.subheader('It is going to be slightly less noisy than usual.')
elif prediction_reg[index_hour] < (time_range_df.loc[index_hour, 'avg'] - time_range_df.loc[index_hour, 'std']):
    st.subheader('It will be quitier than usual.')
else:
    st.subheader('It will be as noisy as usual.')

st.markdown("""---""")



with st.expander('Do you want to see the noise level for the next two days?'):
    # st.header('Noise level with regression model')
    st.plotly_chart(fig_reg)
    st.markdown('This graph shows the absolute levels of noise expected for the next 48 hours in the continuous line, and the avarage of these hours in the dotted line. The color shows the deviation from the average value.')

with st.expander('Do you want to see the weather for the next two days?'):
    st.markdown('<p class="big-font">The weather will be like:</p>', unsafe_allow_html=True)
    st.plotly_chart(weather_fig)
    st.write('The weather forecast is collected from [weather.com](https://weather.com/weather/today/l/634d52f963b8ccca994c4294a53a4a7cb955ef138eb388d37ff579af6f9a4eff)')


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
