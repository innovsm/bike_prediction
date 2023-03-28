import streamlit as st
import pandas as pd
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
# define base model
from keras.models import load_model



@st.cache_data
def get_dataframe_hour():
    data = pd.read_csv("hour.csv")
    
    return data

@st.cache_data
def get_dataframe_day():
    data = pd.read_csv("day.csv")

    
    return data

@st.cache_data
def linear_regression(dataset, timeline):
    data_hour = dataset
    if(timeline == "hour"):
        train_x, test_x, train_y , test_y = train_test_split(data_hour[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
                                                                        'weathersit', 'temp', 'atemp', 'hum', 'windspeed']], data_hour['cnt'], random_state=42)
    else:
        train_x, test_x, train_y , test_y = train_test_split(data_hour[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                                                                        'weathersit', 'temp', 'atemp', 'hum', 'windspeed']], data_hour['cnt'], random_state=42)
        
    normalize  = StandardScaler()
    train_x = normalize.fit_transform(train_x)
    test_x = normalize.fit_transform(test_x)
    model_1 = LinearRegression()
    model_1.fit(train_x, train_y)
    y_pred = model_1.predict(test_x)
    return (r2_score(test_y, y_pred))

@st.cache_data
def random_forest(dataset, timeline):
    data_hour = dataset
    if(timeline == "hour"):
        train_x, test_x, train_y , test_y = train_test_split(data_hour[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
                                                                        'weathersit', 'temp', 'atemp', 'hum', 'windspeed']], data_hour['cnt'], random_state=42)
    else:
        train_x, test_x, train_y , test_y = train_test_split(data_hour[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                                                                        'weathersit', 'temp', 'atemp', 'hum', 'windspeed']], data_hour['cnt'], random_state=42)
        
    normalize  = StandardScaler()
    train_x = normalize.fit_transform(train_x)
    test_x = normalize.fit_transform(test_x)
    model_1 = RandomForestRegressor()
    model_1.fit(train_x, train_y)
    y_pred = model_1.predict(test_x)
    return (r2_score(test_y, y_pred))


@st.cache_data
def decision_tree(dataset, timeline):
    data_hour = dataset
    if(timeline == "hour"):
        train_x, test_x, train_y , test_y = train_test_split(data_hour[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
                                                                        'weathersit', 'temp', 'atemp', 'hum', 'windspeed']], data_hour['cnt'], random_state=42)
    else:
        train_x, test_x, train_y , test_y = train_test_split(data_hour[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                                                                        'weathersit', 'temp', 'atemp', 'hum', 'windspeed']], data_hour['cnt'], random_state=42)
        
    normalize  = StandardScaler()
    train_x = normalize.fit_transform(train_x)
    test_x = normalize.fit_transform(test_x)
    model_1 = DecisionTreeRegressor()
    model_1.fit(train_x, train_y)
    y_pred = model_1.predict(test_x)
    return (r2_score(test_y, y_pred))

@st.cache_data
def svr(dataset, timeline):
    data_hour = dataset
    if(timeline == "hour"):
        train_x, test_x, train_y , test_y = train_test_split(data_hour[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
                                                                        'weathersit', 'temp', 'atemp', 'hum', 'windspeed']], data_hour['cnt'], random_state=42)
    else:
        train_x, test_x, train_y , test_y = train_test_split(data_hour[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                                                                        'weathersit', 'temp', 'atemp', 'hum', 'windspeed']], data_hour['cnt'], random_state=42)
        
    normalize  = StandardScaler()
    train_x = normalize.fit_transform(train_x)
    test_x = normalize.fit_transform(test_x)
    model_1 = SVR(kernel="linear")
    model_1.fit(train_x, train_y)
    y_pred = model_1.predict(test_x)
    return (r2_score(test_y, y_pred))

def app():
    dataset = st.selectbox("Select Dataset", ['','Hourly', 'Daily'])

    model_select  = st.selectbox("Select Mode", ['','LinearRegression', 'DecisionTreeRegressor', 'RandomForestRegressor','SVR',"Deep Learning"])

    if(dataset != ''):
        if(model_select != ''):
            if(model_select == "LinearRegression"):
                if(dataset == 'Hourly'):
                    dataframe = get_dataframe_hour()
                    data = linear_regression(dataframe, "hour")
                    st.write("R2-score")
                    st.write(data)
                elif(dataset == 'Daily'):
                    dataframe = get_dataframe_day()
                    data = linear_regression(dataframe,"day")
                    st.write("R2-score")
                    st.write(data)
            elif(model_select == "RandomForestRegressor"):
                if(dataset == 'Hourly'):
                    dataframe = get_dataframe_hour()
                    data = random_forest(dataframe,'hour')
                    st.write("R2-score")
                    st.write(data)
                elif(dataset == 'Daily'):
                    dataframe = get_dataframe_day()
                    data = random_forest(dataframe,"day")
                    st.write("R2-score")
                    st.write(data)
            elif(model_select == "DecisionTreeRegressor"):
                if(dataset == 'Hourly'):
                    dataframe = get_dataframe_hour()
                    data = decision_tree(dataframe,'hour')
                    st.write("R2-score")
                    st.write(data)
                elif(dataset == 'Daily'):
                    dataframe = get_dataframe_day()
                    data = decision_tree(dataframe,"day")
                    st.write("R2-score")
                    st.write(data)
            elif(model_select == "SVR"):

                if(dataset == 'Hourly'):
                    dataframe = get_dataframe_hour()
                    data = svr(dataframe,'hour')
                    st.write("R2-score")
                    st.write(data)
                elif(dataset == 'Daily'):
                    dataframe = get_dataframe_day()
                    data = svr(dataframe,"day")
                    st.write("R2-score")
                    st.write(data)
            elif(model_select == "Deep Learning"):

                if(dataset == "Hourly"):
                    data_hour = get_dataframe_hour()
                    model = load_model("model.h5")
                    X = data_hour[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday','weathersit', 'temp', 'atemp', 'hum', 'windspeed']]
                    y_pred = model.predict(X)
                    st.write("r2 score")
                    st.write(r2_score(y_pred, data_hour['cnt']))
                 
                else:
                    st.write("do note have enough data to train the neural network")

                
        

                


               


