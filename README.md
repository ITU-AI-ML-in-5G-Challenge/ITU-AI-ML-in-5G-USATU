# ITU-AI-ML-in-5G-USATU

# Traffic classification
## Used for classification network traffic type. This solution used for classification between IOT and Video trafic.

### Requirements
- matplotlib
- pandas
- numpy
- catboost
- sklearn
- imblearn

# Traffic Forecasting
## Used to predict traffic based on previous values
- matplotlib
- pandas
- numpy
- tensorflow
- sklearn

We got the best result when using data from the last 24 hours and one hour exactly one day ago.

We trained two neural network models. One for bytes and one for packets. And got the followingresults:1)Bytes:
RMSE=0.0195
MAPE = 13.43 %
2)Packages:
RMSE = 0.005
MAPE = 16.781 %


