### Ex.No: 6 HOLT WINTERS METHOD


### AIM:
To implement the Holt Winters Method Model using Python.
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative
trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and
Evaluate the model predictions against test data
6. Create the final model and predict future data and plot it

### PROGRAM:
```
DEVELOPED BY:KUKKADAPU CHARAN TEJ
REGISTER NUMBER: 212224040167
```
```py
Importing necessary modules
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
data = pd.read_csv('/content/AirPassengers.csv', parse_dates=['Month'],index_col='Month'
data.head()
data_monthly = data.resample('MS').sum() #Month start
data_monthly.head()
data_monthly.plot()
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten
scaled_data.plot() # The data seems to have additive trend and multiplicative seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()
Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate
the model predictions against test data
Create teh final model and predict future data and plot it
Scaled_data plot:
scaled_data=scaled_data+1 # multiplicative seasonality cant handle non postive values, ye
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()
test_predictions_add = model_add.forecast(steps=len(test_data))
ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')
np.sqrt(mean_squared_error(test_data, test_predictions_add))
np.sqrt(scaled_data.var()),scaled_data.mean()
final_model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_pe
final_predictions = final_model.forecast(steps=int(len(data_monthly)/4)) #for next year
ax=data_monthly.plot()
final_predictions.plot(ax=ax)
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Number of monthly passengers')
ax.set_ylabel('Months')
ax.set_title('Prediction')
```

### OUTPUT:

## Scaled_data plot:
![image](https://github.com/user-attachments/assets/7c93b5b9-1d0c-4941-9c93-a6e0f64d1151)

## Decomposed plot:
![image](https://github.com/user-attachments/assets/2202a94c-d8e2-4d5a-b7c6-e7a354157d5f)

## Test prediction:
![image](https://github.com/user-attachments/assets/df3f68e3-35ad-4092-b002-808ef534c183)

## Model performance metrics:
## RMSE:

![image](https://github.com/user-attachments/assets/fce39285-e890-401f-96d4-7546713c3944)

## Standard deviation and mean:
![image](https://github.com/user-attachments/assets/61f2026d-057d-4b65-bcff-11e02fee178b)

## Final prediction:
![image](https://github.com/user-attachments/assets/46f8691b-32d2-4580-a644-1804efdd6539)

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
