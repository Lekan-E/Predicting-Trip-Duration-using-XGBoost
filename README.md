# Predicting Trip Duration Using Machine Learning
### Project Overview
The CEO of a bike-share company is looking to enhance customer experience by introducing a feature that provides users with an estimated trip duration (ETA) when travelling between stations. By offering an ETA, the company can also estimate the fare for each trip, giving customers a clear cost expectation. From a business perspective, this machine learning (ML) model will be a valuable tool for tracking potential trip revenue.

### Problem Statement
This project focuses on developing an Extreme Gradient Boosting (XGBoost) regression model that can accurately predict ride durations and estimate fares based on multiple influencing factors, such as weather conditions, station distances, user membership, and bike type. By training the model with historical trip data combined with real-time factors, it aims to deliver accurate ETAs for users.

### Key Objectives
1. Data Exploration
2. Machine learning: Build a model to predict trip duration and estimated fare. (We used a similar pricing as BIKE SHARE TORONTO to generate our revenue)
3. Visualization/Implementation: Design an interface that provides users with an estimate of a trip given the start an end station.

![altimage](https://github.com/Lekan-E/Predicting-Trip-Duration-using-XGBoost/blob/4c3ed1f8589de6a8868f2240df839f6fc77af609/Images/cyclistic.jpg)

### Data
The data contains over 5 million records of trips taken from 2020 to 2021. To improve our model's performance, we need to collect/extract the following data:

- Holiday Dates - Via webscraping we extracted the dates and events for the whole year
    - Source: [LINK](https://www.timeanddate.com/weather/usa/chicago/historic?month=1&year=2021)
- Weather: Collect hourly data for each day with over 8,785 records.
    - Source: [LINK](https://www.visualcrossing.com/weather/weather-data-services/Chicago,United%20States/metric/2020-12-01/2021-11-30)

![altimage](https://github.com/Lekan-E/Analysis-for-a-Bike-Sharing-Company-to-Boost-Member-Conversion/blob/f5c1eaeaeaa54350c2ec5ddf7a3f3de858b5de86/Images/Misc/drawSQL-image-export-2024-09-27.png)


## Exploratory Data Analysis
During EDA, we analyzed factors influencing ride durations, such as time of day, day of the week, and month. Weather data provided insights into how temperature, humidity, and wind speed impacted trip times. Below is a simple correlation matrix between our features — 1 being the highest correlation.

![altimage](https://github.com/Lekan-E/Predicting-Trip-Duration-using-XGBoost/blob/4c3ed1f8589de6a8868f2240df839f6fc77af609/Images/correlation.png)

Taking a deeper look at the ride duration by member type, casual riders were observed to have longer trip durations, likely due to leisure or tourism activities, whereas members tended to use the service for commuting. These distinctions will aid in the model’s predictions.

![altimage](https://github.com/Lekan-E/Predicting-Trip-Duration-using-XGBoost/blob/4c3ed1f8589de6a8868f2240df839f6fc77af609/Images/memberdistribution.png)

## Machine Learning Model Overwiew
For the machine learning model selection, we will be dealing with the following two assumptions:
- Supervised Learning - Since we are dealing with labeled data.
- Regression - Since we are dealing with making estimations.

Since we are aiming to predict a continuous target variable, we will use the XGBoost Regression Model. This supervised ML algorithm is particularly effective for large datasets, making it well-suited for accurately predicting trip durations based on a variety of features.

### Steps for Predicting
- **Preprocessing**: Selected features with a high correlation to ride duration based on EDA insights. A sample of 500,000 records was used to train the model, with ride duration as the target variable.

- **Encoding**: Categorical columns (bike type, member type, weather, etc.) were mapped to numerical values.

- **Feature Engineering**: Our first step was to calculate the geographical distance (km) between two the start and end stations, given their positions (latitude and longitude) as we saw in the EDA, this feature was the most important in determining the duration. 

*NOTE. A more accurate approach to calculating the distance is by using an open-source network software that provides us with the most likely route taken along with the distance* 

![altimage](https://github.com/Lekan-E/Predicting-Trip-Duration-using-XGBoost/blob/4c3ed1f8589de6a8868f2240df839f6fc77af609/Images/single-trip.png)

- **Outlier Handling**: Dropping trips that started and ended at the same docking station. This is because if we were to calculate the distance, it would return a distance of 0 km. But in reality, these are trips where the cyclist likely rode to and from the same station. Also, I decided to drop rides running longer than 20 km from each station as these outliers could affect the model performance

![altimage](https://github.com/Lekan-E/Predicting-Trip-Duration-using-XGBoost/blob/4c3ed1f8589de6a8868f2240df839f6fc77af609/Images/duratioin%20distance.png)

- **Hyperparameter Tuning**: Using GridSearch, I tuned the model by evaluating various combinations of hyperparameters to identify the best configuration for achieving the lowest mean absolute error (MAE). This process helped optimize the model's performance by pinpointing the hyperparameters that produced the most accurate results.

## Model Evaluation
The initial attempt at predicting ride durations utilized a random forest model, but it struggled with the large dataset, requiring significantly longer processing times. To address this, the model was switched to XGBoost, which performed more efficiently while providing better accuracy and handling the large dataset more effectively. 

Below is a summary of the XGBoost model's evaluation metrics:
- Mean Absolute Error (MAE): 225.93 seconds
- Mean Squared Error (MSE): 121,746.72
- Root Mean Squared Error (RMSE): 348.92 seconds
- Explained Variance Score: 0.61

The error distribution reveals some insights:
- **Impact on Short and Long Trips**: The XGBoost model had a MAE of 225.93 seconds (approximately 3 minutes, 75 seconds), which equates to about 27% of the average ride duration from all trips in our data. For example, a 3-minute error is more impactful in a 10-minute trip than in a 40-minute one.
- **Effect of Outliers**: The MAE and RMSE show higher variability, likely due to outliers in the dataset, such as longer, unpredictable rides taken by casual users. Casual riders often have more exploratory travel patterns, unlike members who typically follow predictable routes.

## Further Evaluation - Residual Distribution
The residual distribution shown in the histogram provides insights into our ML model's performance and error characteristics. 

![altimage](https://github.com/Lekan-E/Predicting-Trip-Duration-using-XGBoost/blob/4c3ed1f8589de6a8868f2240df839f6fc77af609/Images/residuals.png)

Here are some key observations and conclusions:
- **Centering Around Zero**: The residuals are centered around zero, with a roughly bell-shaped curve, which indicates that, on average, the model does not overestimate or underestimate the ride durations. This is a good sign, as it suggests the model is unbiased.
- **Narrow Peak and Tails**: The narrow peak near zero suggests that the majority of the residuals are close to zero, indicating that the model performs well for most predictions, with small errors. However, there are longer tails, especially on the positive side, which means there are some cases where the model either underestimates or overestimates ride duration. These outliers could result from variations in the data that the model does not capture well, such as unusual ride patterns.

Overall, the residual distribution suggests that the model performs reasonably well with some room for improvement, especially for outlier cases.

## Implementation/Final Results
Below is a look at a sample interactive dashboard that provides a comprehensive overview of bike trips for users. The dashboard displays, the distance between the stations showing the optimal bike route, estimated trip duration and total cost.

Here is a link to the interactive dashboard - [LINK](https://public.tableau.com/views/BikeRideCostEstimatior/RidePlanner?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link).

### Description of the Interactive Elements
- The primary elements is an interface that will allow a user input the start and end station, member and bike type for the machine learning to output an estimated trip duration and fare.
- **Graphhopper** - This allows us to plot the optimal bike route for the trip, given the start and end positions. 

![ALTIMAGE](https://github.com/Lekan-E/Predicting-Trip-Duration-using-XGBoost/blob/4c3ed1f8589de6a8868f2240df839f6fc77af609/Images/Dashboard.png)

Here are some use cases for this dashboard for users and the company:
- **Trip Cost Estimation**: Users can calculate the estimated cost of a bike trip by specifying trip details. This helps members and casual users budget their rides based on distance and bike type.
- **Route Planning**: Users can visualize their intended route on the map, giving them a clear understanding of the trip path.
- **Real-Time Updates**: Commuters can use the dashboard to quickly estimate travel time and cost in real-time, helping them better plan their daily journeys and consider alternative routes if needed.
- **Member Conversion Incentive**: This dashboard provides insights into the value difference between member and casual pricing, making it useful for promoting membership benefits and encouraging casual users to subscribe.
