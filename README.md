#Forecasting Wait Times at the US/Canada Border

## Summary
Travel delays are a frustrating reality of driving and are particularly pronounced when crossing the US/Canada border.  Reliable and accurate predictions of wait times are not available to travelers.  This project predicts the wait times at the Peace Arch and Pacific Highway border crossings and provide users with a more reliable forecast of wait times for dates in 2016.

## Executive Summary
### [Borderforecaster.com](http://borderforecaster.com)
* Web app predicts wait times for Peace Arch and Pacific Highway crossings
* Estimates data for northbound data which is missing data due to sensor issues

### Results
* Model predictions better than day of week averages for last 12 months
* Top features: time of day and weather 
* Holidays that drive wait times are different between travel directions

## Data Source
### JSON API
* [Whatcom Council of Governments](http://www.cascadegatewaydata.com/)
* 20+ million records of wait time and volume
* 5 minute grain
* Since 2007

### Crossings

* Peace Arch
* Pacific Highway
* Sumas
* Lynden

### Lanes

* Car
* Nexus
* Truck
* Bus
* FAST

*This project focused on car lane data at Peace Arch and Pacific Highway crossings only.*

## Goals
### Improve predictions compared to publically available tools
Users can view real-time wait times from [WSDOT](http://www.wsdot.com/traffic/border/) or [US Customs](https://bwt.cbp.gov/), but the data is of limited value to those travelers already near the border.

![](readme_images/wsdot.png)

Alternatively, users can view an average of wait times for a day of week from the [University of California](http://traffic.calit2.net/); however, variations by day of year are disregarded.
![](readme_images/calit.png)

### Provide predictions for northbound crossings
The UC data only provides predictions for southbound crossings.  The reason for the omission is due to gaps in the data due to where the sensors are placed.  According to the data steward, data below a certain threshold are reported as zero.  

The chart below shows volume in red and wait time in blue.  Notice that 12pm and 4pm, even though volume is at a peak, the wait time displays zero.  
![](readme_images/northbound.png)

## Pre-processing
### Imputing false zeros
For northbound data, the false zeros from chart above needed to be imputed before any predictions could be made.  The data was imputed using a decision tree model which used volume and wait time values of neighbors as features.  The imputer consisted of 3 separate decision tree models which were applied depending on whether values from neighbors were available:

* Both lead + lag values
* Lead values only
* Lag values only

Because of the large spans of false zeros, the imputer was applied iteratively, filling in missing values in step-wise fashion.

The imputer was trained on southbound data with data below a configurable threshold set to zero.  Then all values below zero were 

### Smoothing and resampling
Due to the noise in the raw data, data was smoothed with a window size of 1 hour using LOWESS. 

Once smoothed, the data was resampled at 30 minute grain to reduce processing time without reducing the benefit to the end-user.

## Feature Engineering
### Date and time features
For each record, the following date and time features were constructed:

* Time of day
* Year
* Month
* Week
* Day of week

### Holidays
Major holidays from US and Canada were added as features, along with lead and lag effects.  

Lead holiday features were added to account for traveler behavior ahead of a holiday, e.g., the Friday before Labor Day.  Lag holiday features were added to account for traveler behavior after a holiday, e.g. Sunday after Thanksgiving.

### Weather
Weather data was pulled from [Weather Underground](http://www.wunderground.com/) for Blaine, WA using following fields:

* Temperature (min/max/mean)
* Rain/Snow/Thunderstorm/Fog
* Precipitation

Lead and lag weather features were added to account for changes in traveler behavior after a weather event, or in anticipation of a weather event.

### Trend
Wait time has decreased consistently over time as shown in chart below.
![](readme_images/trend.png)

To capture this trend, a difference in daily average wait time was added as a feature.  Multiple differences were added over multiple weeks to capture both long term and short term trends.

### Excluded features

| Feature | Result | 
|---|---|
| School calendars | no improvement |
| Lag daily averages of wait times | overfit |
| Rolling daily averages of wait times | overfit |
| North vs. south volume imbalance | overfit |

## Modeling
### Baseline
A baseline model was defined as the average over the last 12 months by day of week.  The baseline is motivated by the day of week averages referenced above from UC, and by Random Forest models which tended to predict the same values as the baseline model.

Predictions from the baseline model served as measuring stick for comparing the quality of my model.

### Extra Trees
Random Forest was the first model attempted, but was never able to beat the baseline model.  Extra Trees was used instead which yielded better results and more variance in prediction compared to the baseline.  

Once trend features were added to the model, Extra Trees consistently beat the baseline predictions for different crossings, directions and years.

A Gradient Boosting model was tested, but the predictive accuracy was only marginally better than Extra Trees.  The significantly higher processing cost of Gradient Boosting, due to the inability to parallelize model training, favored Extra Trees.

### Ensembling
To further improve the predictive accuracy, the Extra Trees predictions were ensembled with the baseline predictions. Ensembling was performed using a harmonic mean with equal weights.  

Different weights were attempted, but since optimal weights varied depending on the data set (year, crossing and direction), equal weights were used to generalize the model.

### Preventing Overfitting
For any given data set, it was possible to improve the model via hyperparameter tuning.  However, this came at the expense of poorer predictive accuracy for a different data set, i.e. different year.

To keep the model generalizable, the Extra Trees model was loosely tuned with 96 estimators as the only non-default parameter.

### What about ARIMA?
There are a few factors that make ARIMA not applicable:

1. Multiple seasonalities, e.g. daily, weekly and yearly
2. Non-linear exogenous factors
3. Slow to train for large number of exogenous factors

An attempt at using ARIMA yielded predictions that repeated the same seasonal pattern indefinitely.

## Website



## Results

