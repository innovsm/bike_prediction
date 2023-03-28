import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as exp
import matplotlib.pyplot as plt

@st.cache_data
def get_dataframe_hour():
    data = pd.read_csv("hour.csv")
    
    return data

@st.cache_data
def get_dataframe_day():
    data = pd.read_csv("day.csv")

    
    return data

def app():
    final_dataframe = pd.DataFrame() # dummy dataframe
    st.write("EDA [Exploratory data analysis ]")
    data_frame_selection = st.selectbox("Select dataset", ['','hourly dataset', 'daily dataset'])
    if(data_frame_selection == "hourly dataset"):
        data_frame = get_dataframe_hour()
        final_dataframe = data_frame
        
        st.subheader("Dataset overview")
        # printing dataframe metedata
        st.image("dataset_hour.png", width=400)
        st.subheader("Dataset Description")
        st.write(data_frame.describe())
        
    elif(data_frame_selection == "daily dataset"):
        data_frame = get_dataframe_day()
        final_dataframe = data_frame
        st.subheader("Dataset overview")
        st.image("dataset_daily.png", width = 400)
        st.subheader("Dataset Description")
        st.write(data_frame.describe())
    # visulization section
    """
    1> heatmap -- 
    2> line plots -- done
    3> histogram

    """

    st.subheader('Visulization using bar - plot')
    st.write("heatmap provides correlation between attributes")
    button_2 = st.selectbox("select attributes", ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
       'casual', 'registered', 'cnt'])
    if(button_2):
        data = final_dataframe
        try:
            x = data.groupby([button_2]).sum()  # ignore warning
            st.bar_chart(x['cnt'])
        except:
            st.write("please select the dataframe above")

    else:
        st.write("error")
    
    # writing heatmap
    st.subheader("Visulization using heatmap")

    button_heatmap = st.multiselect("Select attributes", ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
       'casual', 'registered', 'cnt'])
    try:
        fig = plt.figure()
        sns.heatmap(final_dataframe[button_heatmap].corr())
        st.pyplot(fig)
    except:
        st.write("please select the dataframe above")

    
    
    
    
    

