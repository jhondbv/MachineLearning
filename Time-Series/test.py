# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
import plotly.express as px

#%% 
# Importando datos
path="../Datasets/store-sales-time-series-forecasting/"
# Load train
train_df = pd.read_csv(path+'train.csv', parse_dates=['date'], infer_datetime_format=True) # columna date a tipo fecha
# Load Test
test_df = pd.read_csv(path+'test.csv', parse_dates=['date'], infer_datetime_format=True) # columna date a tipo fecha
#Load stores
store_df = pd.read_csv(path+'stores.csv')
#Load oil 
oil_df = pd.read_csv(path+'oil.csv',parse_dates=['date'], infer_datetime_format=True)
#Load holidays
holiday_df= pd.read_csv(path+'holidays_events.csv')

#%%
# Separando date en sus componentes dia , mes .año y dia de la semana
train_df['date'] = pd.to_datetime(train_df['date'])
train_df['day_of_week'] = train_df['date'].dt.dayofweek
train_df['month'] = train_df['date'].dt.month
train_df['year'] = train_df['date'].dt.year

test_df['date'] = pd.to_datetime(test_df['date'])
test_df['day_of_week'] = test_df['date'].dt.dayofweek
test_df['month'] = test_df['date'].dt.month
test_df['year'] = test_df['date'].dt.year


#%% Separar train data en datos de entrenamiento y test
train_data = train_df[train_df["date"]<="2017-03-01"]
test_data = train_df[train_df["date"]>"2017-03-01"]
print(train_df.info())
print(test_df.info())


#%%
#Grafica de la media de ventas en train y test 
train_data_mean = train_data[['date','sales']].groupby(['date']).mean()
test_data_mean = test_data[['date','sales']].groupby(['date']).mean()

#%%
import plotly.graph_objs as go
train_values= train_data_mean.copy()
train_values=train_values.iloc[1:, :].reset_index()
test_values= test_data_mean.copy()
test_values=test_data_mean.iloc[1:, :].reset_index()

fig = go.Figure()

main_trace =  go.Scatter(
    x=train_values['date'],
    y=train_values['sales'],
    mode='lines',
    name='train'
)

new_trace = go.Scatter(
    x=test_values['date'],
    y=test_values['sales'],
    mode='lines',
    name='test'
)
fig.add_trace(main_trace)
fig.add_trace(new_trace)

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

fig.show()

#%%

# Merge 
# Merge de train y test con stores
train_data = pd.merge(train_data, stores_data, on='store_nbr', how='left')
test_data = pd.merge(test_data, stores_data, on='store_nbr', how='left')


# Codificando variables categoricas
train_data = pd.get_dummies(train_data, columns=['family', 'type', 'city', 'state', 'cluster'])
test_data = pd.get_dummies(test_data, columns=['family', 'type', 'city', 'state', 'cluster'])
# nan data validation
test_data.isna().any().any()

pd.set_option('display.max_columns', 1000)
train_data.info()

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data.drop(['date', 'sales'], axis=1)
                                                  , train_data['sales']
                                                  , test_size=0.2, random_state=42)
X_train.shape, X_val.shape, y_train.shape, y_val.shape

# Train a Random Forest Regression model
model = RandomForestRegressor(n_estimators=50,max_depth=5, random_state=42)

#%%
model.fit(X_train, y_train)

#%%
# Make predictions on the validation set
y_val_pred = model.predict(X_val)

# Calculate the RMSLE of the predictions
rmsle_score = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(y_val_pred)))
print('RMSLE: ', rmsle_score)