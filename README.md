# Challenge: power curve modelling benchmarking (WeDoWind)

## Data:
The script works with the Kelmarsh and Penmanshiel site data provided by the challenge organizers. Contact me for more info on the training dataset.

## Scripts:
The reading and writing part of the files is based on the ANN code uploaded by Arthur Girard on Gitlab (thanks!). https://gitlab.ethz.ch/arthurgirard/datadriveturbineperformanceanalysis/-/blob/main/Simple_ANN_CustomLoss_commented.py 

## Data pre-processing:
For all 4 solutions, DBSCAN was used for automatic outlier detection and removal from the training database. It is a density-based clustering algorithm. The points lying in the low-density region- due to hysteresis effects, averaging or measurement errors are automatically detected and removed from the training database.

DBSCAN from the sklearn library in Python was used with the following settings:
```
db = DBSCAN(eps=0.2, min_samples=10).fit(df[["wind_speed", "power"]])
```
These settings were selected on the basis of trial and error in a few cases to make sure that useful measurements near the cut-out wind speed are not eliminated. Lowering the number of min_samples can lead to the formation of more than two clusters- we ideally only want two clusters- one of the clean data and the other of the noise.

## ML model: 
XGBoost (extreme gradient boosting) algorithm was used in all 4 cases. This choice was based on the fact that XGBoost is a very flexible, non-parametric approach, that naturally accounts for heteroscedastic responses. The differences in the 4 submissions are listed in the following section.

## Submissions:
### 1: XGBoost with default settings + all features
The model was trained on the following features: 
'wind_speed', 'wind_speed_sensor1', 'wind_speed_sensor1_SD', 'wind_speed_sensor2', 'wind_speed_sensor2_SD', 'density_adjusted_wind_speed', 'wind_direction', 'nacelle_position', 'wind_direction_SD', 'nacelle_position_SD', 'nacelle_ambient_temperature', 'TI','Day.Night'
The default parameter settings of the model can be found here: https://xgboost.readthedocs.io/en/stable/python/python_api.html

```
def xgb_regressor(x, y, x_test):
    model = xgb.XGBRegressor()
    model.fit(x, y)
    y_hat = model.predict(x_test)
    return(y_hat)
```

### 2: XGBoost with default settings + only wind features
The model was trained on the following features: 
'wind_speed', 'wind_speed_sensor1', 'wind_speed_sensor1_SD', 'wind_speed_sensor2', 'wind_speed_sensor2_SD', 'density_adjusted_wind_speed'
Based on the feature analysis, it seemed that the wind speed-related features had the highest importance, and therefore this model was trained only on the features listed above.
The default parameter settings of the model can be found here: https://xgboost.readthedocs.io/en/stable/python/python_api.html

### 3: XGBoost with custom loss function and early-stopping + all features
The model was trained on the following features: 
'wind_speed', 'wind_speed_sensor1', 'wind_speed_sensor1_SD', 'wind_speed_sensor2', 'wind_speed_sensor2_SD', 'density_adjusted_wind_speed', 'wind_direction', 'nacelle_position', 'wind_direction_SD', 'nacelle_position_SD', 'nacelle_ambient_temperature', 'TI','Day.Night'

A custom loss function was defined based on a combination of RMSE and pseudo-huber loss (in lieu of MAE). An early-stopping algorithm based on the RMSE of a cross-validation set was introduced, with early stopping rounds fixed to 100.

def xgb_regressor_custom_loss(x, y, x_test, x_val, y_val):
    def custom_loss(y_pred, y_val):
        d = (y_val-y_pred)
        delta = 1  
        scale = 1 + (d / delta) ** 2
        scale_sqrt = np.sqrt(scale)
        grad = d / scale_sqrt + 2 * d
        hess = (1 / scale) / scale_sqrt + 2
        return grad, hess
    
    model = xgb.XGBRegressor(n_estimators=5000,
                       early_stopping_rounds=100,
                       objective = custom_loss,
                       max_depth=3,
                       eval_metric = 'rmse',
                       learning_rate=0.01)
    model.fit(x, y,
        eval_set=[(x, y), (x_val, y_val)], verbose = 500)

    y_hat = model.predict(x_test)
    return(y_hat)

### 4: XGBoost with custom loss function and early-stopping  + only wind features
The model was trained on the following features: 
'wind_speed', 'wind_speed_sensor1', 'wind_speed_sensor1_SD', 'wind_speed_sensor2', 'wind_speed_sensor2_SD', 'density_adjusted_wind_speed'

The custom loss function is the same as in case #3
