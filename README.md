# Stock Market Analysis

## Overview

This project involves the analysis and prediction of stock prices using various time series forecasting techniques. The primary goal is to forecast the stock prices of a specific company (e.g., Reliance Industries) for the next 30 days. The analysis covers different methods, including Single Exponential Smoothing, Holts Winter methods, ARIMA, and SARIMAX.

## Data Collection

The stock price data was collected using the Yahoo Finance API for a specific company, in this case, Reliance Industries.

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the dataset's structure, visualize trends, and identify patterns.

## Univariate Analysis

Univariate analysis was conducted on various stock price attributes, including Open, Close, High, Low, and Volume.

## Time Series Forecasting Models

### Single Exponential Smoothing

A simple exponential smoothing model was applied to analyze and forecast stock prices.

### Holts Winter Methods

Different variants of the Holts Winter exponential smoothing model were implemented, considering additive and multiplicative seasonality with additive and multiplicative trends.

### ARIMA

An AutoRegressive Integrated Moving Average (ARIMA) model was applied for time series forecasting.

### SARIMAX

The Seasonal AutoRegressive Integrated Moving Average with eXogenous factors (SARIMAX) model was implemented for improved forecasting, considering seasonality and external factors.

## Model Comparison

The performance of each model was evaluated and compared based on Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

## Final Forecast

The final forecast was made using the most accurate model, and the results were visualized for the next 30 days.

## Deployment on Binder

The Jupyter Notebook for this project can be interactively explored on Binder. Click the following link to access the notebook:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RohitGaikwad1/Stock-Market-Analysis/5ce314e9aa0cf13062d4d3cc666127341a2fc273?urlpath=lab%2Ftree%2FCode%20and%20presentation%2FReliance%20Stock%20Prediction%20Final%20Project.ipynb)

## Conclusion

The project provides insights into time series forecasting techniques for stock prices, demonstrating the strengths and weaknesses of different models. The final forecast can be used for informed decision-making in financial scenarios.

Feel free to clone and explore the Jupyter Notebook for a detailed analysis.
