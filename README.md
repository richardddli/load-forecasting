# load-forecasting
This load forecaster is designed to perform short-term & long-term load forecasting for a 24-hour period.

# Data Source
15 min interval load data and hourly temperature data is available [**here**](load_temperature_data.csv) from November 2012 to December 2013 for a sample household in California.

# Model Selection
An Multi-layer Perceptron regressor was selected to model this load data, in order to accommodate the nonlinear interactions of load demand. Eight predictor variables were implemented:

* interpolated temperature \*
* 24 hr lagged load \*
* 7 day lagged load \*
* previous day average load \*
* time of day
* day of week
* day of year
* weekend/holiday

\* used for short-term forecasting only

# Forecasting a 24 hr period
You can train the model and perform forecasting over a 24-hr period. The `forecast_date` indicates the start time of the 24-hr interval.

```python
import forecaster
forecaster.train_model_and_forecast(load_data='load_temperature_data.csv', forecast_date='2018-6-30 14:30', 
                                    holidays='US Bank holidays.csv', short_term=False)
```

You can also perform cross-validation using a time series split, to evaluate the performance of the model:
```python
import forecast_evaluator
forecast_evaluator.cross_validate(load_data='load_temperature_data.csv', 
                                  holidays='US Bank holidays.csv', short_term=False)
```

# Notebook
The [documentation](Methodology%20and%20Key%20Findings.docx) is currently undergoing conversion to a Jupyter notebook.
