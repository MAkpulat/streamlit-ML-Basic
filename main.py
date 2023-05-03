

import streamlit as st 
import pandas as pd 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()



with header:
    st.title('selam ben Murat AKPULAT')
    st.text('here, i look into the transactions of taxi')

with dataset: 
    st.header('NYC taxi dataset')
    st.text('dataset will be here')

    taxi_data = pd.read_csv('data/taxi_data.csv')
    taxi_data = taxi_data.loc[1:100000,:]
    st.write(taxi_data.shape)
    st.write(taxi_data.head())


    local_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts().head(50))
    st.bar_chart(local_dist)



with features:
    st.header('the features i create')
    st.text('Features will be extracted')

with model_training:
    st.header('time to train modell !!!!')
    st.text('train model with different methods')

    sel_col, disp_col = st.columns(2)


    max_depth = sel_col.slider('what should be ma depth of the model?' , min_value=10, max_value=100, value = 20, step = 10 )
    n_estimators = sel_col.selectbox('how many trees should there? ', options=[100,200,300,'no limit'], index=0)
    input_feature = sel_col.text_input('which feature should be used as input feature ? ', 'PULocationID')

    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]

    regr.fit(X,y)
    prediction = regr.predict(y)

    disp_col.subheader('MAE')
    disp_col.write(mean_absolute_error(y,prediction))

    disp_col.subheader('MSE')
    disp_col.write(mean_squared_error(y,prediction))

    disp_col.subheader('R2')
    disp_col.write(r2_score(y,prediction))
    