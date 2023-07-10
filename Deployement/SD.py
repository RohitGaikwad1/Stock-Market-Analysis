#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import time
from datetime import datetime, date
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandas.tseries.offsets import DateOffset
import base64
from streamlit_lottie import st_lottie
import requests


# In[2]:


st.header("Stock Market Analysis")


# In[3]:


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None 
    return r.json()

animation_url = "https://assets9.lottiefiles.com/packages/lf20_oat3z2uf.json"
lottie_data = load_lottieurl(animation_url)


# In[4]:


st_lottie(lottie_data, height=300,width=300)


# In[5]:


ticker_options = ['RELIANCE.NS', 'AAPL', 'GOOGL']
default_ticker = ticker_options[0]
ticker = st.selectbox('Select Ticker', ticker_options, index=0)
start_date = st.date_input('Start Date', date(2015, 12, 1))
end_date = st.date_input('End Date', date(2022, 12, 31))
period1 = int(time.mktime(start_date.timetuple()))
period2 = int(time.mktime(end_date.timetuple()))
interval = '1d'  # You can change this to '1m' if needed
query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
df = pd.read_csv(query_string)
st.dataframe(df)
csv_filename = f'{ticker}.csv'
df.to_csv(csv_filename, index=False)
st.success(f'Data saved to {csv_filename}')

Reliance=df.drop(["Adj Close"],axis=1).reset_index(drop=True)

Reliance['Date']=pd.to_datetime(Reliance['Date'],format='%Y-%m-%d')

Reliance4 = Reliance.copy()
Reliance4 = Reliance4.set_index(['Date'])


# In[6]:


st.set_option('deprecation.showPyplotGlobalUse', False)
def plot_visualizations(data):
    st.title("Reliance Stock Visualizations")
    st.sidebar.title("Visualization Options")
    
    columns = st.sidebar.multiselect('Select Column',data.columns)
    st.sidebar.write("You selected", len(columns), 'columns')
    selected_option = st.sidebar.radio('Select Plots type', ['Line Plots', 'Box Plots', 'Histograms', 'KDE Plots', 'Heatmap', 'Volume Plot'])

    if selected_option == 'Line Plots':
        for column in columns:
            st.markdown('#### Line plots')
            plt.figure(figsize=(20,10))
            plt.plot(data[column], color='green')
            plt.xlabel('Date')
            plt.ylabel('Open Price')
            plt.title('Open')
            st.pyplot()
    elif selected_option == 'Box Plots':
        for column in columns:
            st.markdown('#### Box plots')
            plt.figure(figsize=(20,10))
            plt.boxplot(data[column])
            plt.xlabel('Date')
            plt.ylabel('Open Price')
            plt.title('Open')
            st.pyplot()
    elif selected_option == 'Histograms':
        st.markdown('#### Histograms')
        for column in columns:
            
            plt.figure(figsize=(20,18))
            plt.hist(data[column], bins=50, color='green')
            plt.xlabel("Open Price")
            plt.ylabel("Frequency")
            plt.title('Open')
            st.pyplot()
    elif selected_option == 'KDE Plots':
        for column in columns:
            st.markdown('#### KDE plots')
            plt.figure(figsize=(20,10))
            sns.kdeplot(data[column], color='green')
            plt.title('Open')
            st.pyplot()
    elif selected_option == 'Heatmap':
            st.markdown('#### Heatmap')
            plt.figure(figsize=(10,8))
            sns.heatmap(data.corr(), annot=True)
            st.pyplot()
    elif selected_option == 'Volume Plot':
        for column in columns:
            st.markdown('#### Volume')
            plt.figure(figsize=(30,10))
            plt.plot(data[column])
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.title('Date vs Volume')
            st.pyplot()



plot_visualizations(Reliance4)


# In[7]:


columns = ['Open','High','Low','Close','Volume']
st.sidebar.title("Moving Average")
st.title("Moving Average")
column = st.sidebar.radio("Select Any Column : ", options =columns)
number = st.sidebar.slider("Enter Rolling Mean : ",min_value = 1, max_value = 300)
ma = Reliance[column].rolling(number).mean()
ma_plot = plt.figure(figsize = (12,6))
plt.plot(Reliance[column])
plt.plot(ma, 'red')
st.pyplot(ma_plot)


# In[8]:


def visualize_reliance_data(data):
    st.title("Yearly Sum Of The Volume")
    st.sidebar.title("Yearly Sum Of The Volume")
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month

    groupby_year = data.groupby(['Year']).sum()
    
    chart_type = st.sidebar.radio('Select chart type', ['Bar Chart', 'Pie Chart', 'Line Chart'])
    
    if chart_type == 'Bar Chart':
        ax = groupby_year['Volume'].plot(kind='bar', figsize=(12, 6), edgecolor='black')
        plt.xlabel('Year', fontsize=14)

        for i in ax.containers:
            ax.bar_label(i)

    elif chart_type == 'Pie Chart':
        groupby_year['Volume'].plot(kind='pie', figsize=(8, 8), explode=[0, 0, 0, 0, 0, 0.05, 0, 0], autopct='%1.2f%%')
        plt.title('Volume', fontsize=14)

    elif chart_type == 'Line Chart':
        groupby_year['Volume'].plot(figsize=(10, 6), color='Skyblue', marker='X')

    st.pyplot()


visualize_reliance_data(Reliance)


# In[9]:


import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

st.set_option('deprecation.showPyplotGlobalUse', False)

def evaluate_holt_winter(data):
    st.sidebar.title("Models Options")
    st.title("Models")
    st.markdown('###### The length of dataset - The input given value = Test Size , eg : 1751-1651 = 100')
    train_slider = st.sidebar.number_input('Train Set Size', min_value=0, max_value=len(data))
    test_slider = st.sidebar.number_input('Test Set Size', min_value=0, max_value=500)

    Train = data.iloc[:train_slider].dropna()
    Test = data.iloc[train_slider:train_slider + test_slider].dropna()

    columns = st.sidebar.multiselect('Select Column', data.columns)
    st.sidebar.write("You selected", len(columns), 'columns')
    model_type = [
        'Holts Method',
        'Holts Winter ExponentialSmoothing additive Seasonality and additive Trend',
        'Holts Winter ExponentialSmoothing Multiplicative Seasonality and additive Trend',
        'Holts Winter ExponentialSmoothing Multiplicative Seasonality and Multiplicative Trend'
    ]

    mape_values = []

    for model_name in model_type:
        if model_name == 'Holts Method':
            st.markdown('#### Holt method')
            for column1 in columns:
                hw_model = Holt(Train[column1]).fit(smoothing_level=0.3, smoothing_slope=0.3)
                pred_hw = hw_model.predict(start=Test.index[0], end=Test.index[-1])
                MAPE_hw_model = mean_absolute_percentage_error(pred_hw, Test[column1])
                st.write("MAPE_hw_model:", MAPE_hw_model)
                rmse_hw_model = mean_squared_error(pred_hw, Test[column1])
                st.write("rmse_hw_model:", rmse_hw_model)
                mape_values.append(MAPE_hw_model)

                plt.figure(figsize=(12, 5), dpi=100)
                plt.plot(Train[column1], label='training')
                plt.plot(Test[column1], label='actual')
                plt.plot(pred_hw, label='forecast')
                plt.title('Forecast vs Actual')
                plt.legend(loc='upper left', fontsize=8)
                st.pyplot()

        elif model_name == 'Holts Winter ExponentialSmoothing additive Seasonality and additive Trend':
            st.markdown('#### Holts Winter ExponentialSmoothing additive Seasonality and additive Trend')
            for column1 in columns:
                hwe_model_add_add = ExponentialSmoothing(
                    Train[column1], seasonal='add', trend='add', seasonal_periods=test_slider
                ).fit()
                pred_hwe_add_add = hwe_model_add_add.predict(start=Test.index[0], end=Test.index[-1])
                MAPE_hwe_add_add_model = mean_absolute_percentage_error(pred_hwe_add_add, Test[column1])
                st.write("MAPE_hwe_add_add_model:", MAPE_hwe_add_add_model)
                rmse_hwe_add_add_model = mean_squared_error(pred_hwe_add_add, Test[column1])
                st.write("rmse_hwe_add_add_model:", rmse_hwe_add_add_model)
                mape_values.append(MAPE_hwe_add_add_model)

                plt.figure(figsize=(12, 5), dpi=100)
                plt.plot(Train[column1], label='training')
                plt.plot(Test[column1], label='actual')
                plt.plot(pred_hwe_add_add, label='forecast')
                plt.title('Forecast vs Actual')
                plt.legend(loc='upper left', fontsize=8)
                st.pyplot()

        elif model_name == 'Holts Winter ExponentialSmoothing Multiplicative Seasonality and additive Trend':
            st.markdown('#### Holts Winter ExponentialSmoothing Multiplicative Seasonality and additive Trend')
            for column1 in columns:
                hwe_model_mul_add = ExponentialSmoothing(
                    Train[column1], seasonal='mul', trend='add', seasonal_periods=test_slider
                ).fit()
                pred_hwe_mul_add = hwe_model_mul_add.predict(start=Test.index[0], end=Test.index[-1])
                MAPE_hwe_model_mul_add_model = mean_absolute_percentage_error(pred_hwe_mul_add, Test[column1])
                st.write("MAPE_hwe_model_mul_add_model:", MAPE_hwe_model_mul_add_model)
                rmse_hwe_model_mul_add_model = mean_squared_error(pred_hwe_mul_add, Test[column1])
                st.write("rmse_hwe_model_mul_add_model:", rmse_hwe_model_mul_add_model)
                mape_values.append(MAPE_hwe_model_mul_add_model)

                plt.figure(figsize=(12, 5), dpi=100)
                plt.plot(Train[column1], label='training')
                plt.plot(Test[column1], label='actual')
                plt.plot(pred_hwe_mul_add, label='forecast')
                plt.title('Forecast vs Actual')
                plt.legend(loc='upper left', fontsize=8)
                st.pyplot()

        elif model_name == 'Holts Winter ExponentialSmoothing Multiplicative Seasonality and Multiplicative Trend':
            st.markdown('#### Holts Winter ExponentialSmoothing Multiplicative Seasonality and Multiplicative Trend')
            for column1 in columns:
                hwe_model_mul_mul = ExponentialSmoothing(
                    Train[column1], seasonal='mul', trend='mul', seasonal_periods=test_slider
                ).fit()
                pred_hwe_mul_mul = hwe_model_mul_mul.predict(start=Test.index[0], end=Test.index[-1])
                MAPE_hwe_model_mul_mul_model = mean_absolute_percentage_error(pred_hwe_mul_mul, Test[column1])
                st.write("MAPE_hwe_model_mul_mul_model:", MAPE_hwe_model_mul_mul_model)
                rmse_hwe_model_mul_mul_model = mean_squared_error(pred_hwe_mul_mul, Test[column1])
                st.write("rmse_hwe_model_mul_mul_model:", rmse_hwe_model_mul_mul_model)
                mape_values.append(MAPE_hwe_model_mul_mul_model)

                plt.figure(figsize=(12, 5), dpi=100)
                plt.plot(Train[column1], label='training')
                plt.plot(Test[column1], label='actual')
                plt.plot(pred_hwe_mul_mul, label='forecast')
                plt.title('Forecast vs Actual')
                plt.legend(loc='upper left', fontsize=8)
                st.pyplot()

    compare_button = st.button("Compare Models")
    threshold = st.number_input("Threshold", value=0.1)

    if compare_button:
        best_model_index = np.argmin(mape_values)
        best_model_mape = mape_values[best_model_index]

        if best_model_mape < threshold:
            best_model_name = model_type[best_model_index]
            st.write(f"The best model is {best_model_name} with MAPE: {best_model_mape:.4f}")
            
    model_type = st.selectbox("Select the forecasting model type:", ['Holts Method',
                                                                         'Holts Winter ExponentialSmoothing additive Seasonality and additive Trend',
                                                                         'Holts Winter ExponentialSmoothing Multiplicative Seasonality and additive Trend',
                                                                         'Holts Winter ExponentialSmoothing Multiplicative Seasonality and Multiplicative Trend'])    
    number = st.slider("Enter the number of forecasting day : ", min_value=1, max_value=365)       
    if st.button('Generate Plot'):
        if model_type == 'Holts Method':
            model = hw_model
        elif model_type == 'Holts Winter ExponentialSmoothing additive Seasonality and additive Trend':
            model = hwe_model_add_add
        elif model_type == 'Holts Winter ExponentialSmoothing Multiplicative Seasonality and additive Trend':
            model = hwe_model_mul_add
        elif model_type == 'Holts Winter ExponentialSmoothing Multiplicative Seasonality and Multiplicative Trend':
            model = hwe_model_mul_mul

        forecast = model.forecast(number)
        st.write(forecast)
        final_plot = plt.figure(figsize=(12, 6))
        plt.plot(data[column1], label='Actual')
        plt.plot(forecast, label='Forecast')
        plt.legend(loc='upper left', fontsize=12)
        st.pyplot(final_plot)



evaluate_holt_winter(Reliance)


# In[ ]:




