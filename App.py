import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from selenium import webdriver
import pandas as pd
from bs4 import BeautifulSoup
driver = webdriver.Chrome("~/Downloads/chromedriver")
driver.get("https://weather.com/weather/hourbyhour/l/c097b546627cdff2da1e276cb9b2731055718a5e7270d777a92857a9701c7870")
content = driver.page_source
soup = BeautifulSoup(content)
temp_val = soup.findAll('span', attrs={'class':'DetailsSummary--tempValue--jEiXE'})
hour = soup.findAll('h3', attrs={'class':'DetailsSummary--daypartName--kbngc'})
wind = soup.findAll('span', attrs={'class':'Wind--windWrapper--3Ly7c DetailsTable--value--2YD0-'})



weather = pd.read_csv('meteo_data/data_final_meteo.csv')
st.title("Noise forecast")

with st.sidebar:
    st.header('Calendar')
    st.header('Weather')

d = st.date_input(
    "Select a date",
    datetime.date(2019, 7, 6))


st.header('Weather')
st.dataframe(weather)