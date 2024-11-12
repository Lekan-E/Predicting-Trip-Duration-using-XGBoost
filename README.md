# Predicting-Trip-Duration-using-XGBoost
### Project Overview
The CEO of a bike-share company is looking to enhance customer experience by introducing a feature that provides users with an estimated trip duration (ETA) when travelling between stations. By offering an ETA, the company can also estimate the fare for each trip, giving customers a clear cost expectation. From a business perspective, this machine learning (ML) model will be a valuable tool for tracking potential trip revenue.

### Problem Statement
This project focuses on developing an Extreme Gradient Boosting (XGBoost) regression model with that can accurately predict ride durations and estimate fares based on multiple influencing factors, such as weather conditions, station distances, user membership, and bike type. By training the model with historical trip data combined with real-time factors, it aims to deliver accurate ETAs for users.

### Key Objectives
1. Data Exploration & Visualization
2. Machine learning:
    - Build a model to predict trip duration and estimated fare. (We used a similar pricing as BIKE SHARE TORONTO to generate our revenue.)

### Data
The data contains over 5 million records of trips taken from 2020 to 2021. To improve our model's performance, we need to collect/extract the following data:

- Holiday Dates - Via webscraping we extracted the dates and events for the whole year
    - Source: https://www.timeanddate.com/weather/usa/chicago/historic?month=1&year=2021
- Weather: Collect hourly data for each day with over 8,785 records.
    - Source: (PAID) https://www.visualcrossing.com/weather/weather-data-services/Chicago,United%20States/metric/2020-12-01/2021-11-30

![altimage](https://github.com/Lekan-E/Analysis-for-a-Bike-Sharing-Company-to-Boost-Member-Conversion/blob/f5c1eaeaeaa54350c2ec5ddf7a3f3de858b5de86/Images/Misc/drawSQL-image-export-2024-09-27.png)


## Exploratory Data Analysis
During EDA, we analyzed factors influencing ride durations, such as time of day, day of the week, and month. Weather data provided insights into how temperature, humidity, and wind speed impacted trip times. Below is a simple correlation matrix between our features — 1 being the highest correlation.