#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:22:16 2021

@author: julian
"""



#%%
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess


#%%
df = pd.read_csv(r'df_sum_ports.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
#df_ts = df.set_index('')
series = df['Energy (kWh)']


#%%
df_X_train = np.load('X_all_30_train.npy')
df_y_test = np.load('y_all_30_test.npy')

#%%
X_all_30_train = np.load('X_all_30_train.npy')
X_all_30_test = np.load('X_all_30_test_new.npy')
X_all_7_train = np.load('X_all_7_train.npy')
X_all_7_test = np.load('X_all_7_test_new.npy')
y_all_30_train = np.load('y_all_30_train.npy')
y_all_30_test = np.load('y_all_30_test_new.npy')
y_all_7_train = np.load('y_all_7_train.npy')
y_all_7_test = np.load('y_all_7_test_new.npy')



#%%
# fit model
model = ARIMA(series, order=(30,1,0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
print(residuals.describe())

#%% 
x = X_all_30_test[0,:]
y = y_all_30_test[0,:]

#%%
fig, ax = plt.subplots(2)

ax[0].plot(np.append(x,y))
#ax[1].plot(df_y_test.flatten())


#%%
history = X_all_30_test[0,:]
model = ARIMA(history, order=(30,0,0))
model_fit = model.fit() 

#%%
parameters = model_fit.polynomial_ar

#%%
output = model_fit.get_forecast(steps = 30) 
yhat = output.predicted_mean
PIs = output.conf_int()
stds = model_fit.cov_params()

#%%
print(model_fit.summary())
              
#%%
np.savetxt('parameters.txt', parameters)

#%%
ma_coefs = np.loadtxt('ma_terms.txt')
ma_coefs_v2 = np.loadtxt('ma_terms_R.txt')

to_add_ma = np.cumsum(ma_coefs**2)
to_add_ma[0] = 0
to_add_ma_R = np.cumsum(ma_coefs_v2**2)
to_add_ma_R[0] = 0
#%%
sigma2 = 4206.733
sigmah = sigma2*(1 + to_add_ma_R)

conf_int_upper = yhat + 1.96*np.sqrt(sigmah)
conf_int_lower = yhat - 1.96*np.sqrt(sigmah)

#%%
plt.plot(yhat,'red')
plt.plot(conf_int_upper,'r--')
plt.plot(conf_int_lower,'r--')
plt.plot(y_all_30_test[0].flatten())

#%%
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


#%%
history_30 = 0
predictions_30 = []
window_size_30 = 30
CIs_30 = []
params_30 = []
res_errors_30 = []
yhat_30 = []


#%%
# walk-forward validation
for t in range(len(X_all_30_test)):
    history_30 = X_all_30_test[t,:]
    model = ARIMA(history_30, order=(30,0,0))
    model_fit = model.fit() 
    output = model_fit.get_forecast(steps = window_size_30) 
    
    yhat_30.append(output.predicted_mean)
    CIs_30.append(output.conf_int())
    predictions_30.append(output.predicted_mean)
    obs = y_all_30_test[t,:]
    params_30.append(model_fit.polynomial_ar)
    res_errors_30.append(yhat_30[t]-obs)
    
    #history.append(obs) 
    #print('predicted=%f, expected=%f' % (yhat, obs))
    
#%% RMSE
yhat_30 = np.array(yhat_30)
rmse_30 = rmse()
print(rmse_30)

#%% 
yhat_30 = np.array(yhat_30)

#%%
plt.plot(yhat_30[3,:],'red')
plt.plot(y_all_30_test[3,:])

#%%
np.save('predictions_AR_30days.npy', yhat_30.flatten())


#%%
history_7 = 0
predictions_7 = []
window_size_7 = 7
CIs_7 = []
params_7 = []
res_errors_7 = []
yhat_7 = []

#%%
for t in range(len(X_all_7_test)):
    history_7 = X_all_7_test[t,:]
    model = ARIMA(history_7, order=(7,0,0))
    model_fit = model.fit() 
    output = model_fit.get_forecast(steps = window_size_7) 
    
    yhat_7.append(output.predicted_mean)
    CIs_7.append(output.conf_int())
    predictions_7.append(output.predicted_mean)
    obs = y_all_7_test[t,:]
    params_7.append(model_fit.polynomial_ar)
    res_errors_7.append(yhat_7[t]-obs)
    
    #history.append(obs) 
    #print('predicted=%f, expected=%f' % (yhat, obs))
    
#%% RMSE
yhat_7 = np.array(yhat_7)
rmse_7 = np.sqrt((np.sum((yhat_7.flatten()-y_all_7_test.flatten())**2))/len(yhat_7.flatten()))
print(rmse_7)

#%%
#print(np.sqrt(((yhat_7.flatten() - y_all_7_test.flatten()) ** 2).mean()))
np.save('predictions_AR_7days_new_v3.npy', yhat_7)

#%%
loaded_pred=np.load('predictions_AR_7days_new_v3.npy')

#%%  Plotting with prediction intervals





#%%
flat_list = [item for sublist in predictions for item in sublist]

#%%
fig, ax = plt.subplots()
ax.plot(flat_list)
ax.plot(y_all_30_test.flatten(), 'red')

#%%
idx_7 = np.arange(0,len(yhat_7.flatten()), 7)
yhat_to_plot = (yhat_7.flatten())[idx_7]
y_all_to_plot = (y_all_7_test.flatten())[idx_7]


#for i in range(0,len(predictions_7), 7):
    


#%%
fig, ax = plt.subplots()
ax.plot(yhat_to_plot, 'red')
ax.plot(y_all_to_plot)

#%%
fig, ax = plt.subplots()
ax.plot(y_all_7_test[20], 'red')
ax.plot(predictions_7[20])

#%%




#%%

model_fit.plot_predict(90, 120, dynamic=True, ax=ax, plot_insample=False)



#%%
X = series.values
size = 2713
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

#%%
model = ARIMA(df_X_train[-1], order=(30,0,0))
model_fit = model.fit()

#%%
forecast = model_fit.predict()
#yhat = forecast.predicted_mean
#yhat_conf_int = forecast.conf_int(alpha=0.05)
#CI = conf_int
#obs = test[:forecast_window]

#%%
plt.plot(df_X_train[-1])
plt.plot(forecast, color='red')
plt.show()

#%%
forecast_window = 100
forecast = model_fit.get_forecast(forecast_window)
yhat = forecast.predicted_mean
yhat_conf_int = forecast.conf_int(alpha=0.05)
#CI = conf_int
obs = df_X_train[-1][:forecast_window]

#%%
rmse = sqrt(mean_squared_error(obs, yhat))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.plot(obs)
plt.plot(yhat, color='red')
plt.plot(yhat_conf_int[:,0],'--r', alpha=0.6)
plt.plot(yhat_conf_int[:,1],'--r', alpha=0.6)
plt.show()

