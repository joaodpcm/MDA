import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

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