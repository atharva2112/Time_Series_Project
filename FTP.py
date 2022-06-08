#%%
from tracemalloc import start
from scipy.stats import chi2
from cProfile import label
from multiprocessing.spawn import import_main_path
from unicodedata import category, name
from importlib_metadata import distribution
import pandas as pd
# from Lab4 import X_test, X_train
import numpy as np
from pyparsing import col
import toolbox as tx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import math
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
import statsmodels.api as sm
from scipy import signal
# %%
def difference(dataset, interval =1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i]- dataset[i-interval]
        diff.append(value)
    return diff

def log_trans(column):
    lg= []
    for i in range(len(column)):
        value = math.log(column[i])
        lg.append(value)
    return lg
#%%
#%%
df  = pd.read_csv("/Users/atharvah/GWU/Untitled/personal/Time_Series/Labs/D202.csv")
df.DATE = pd.to_datetime(df.DATE + ' ' + df["END TIME"])
df = df.drop(columns=["START TIME","END TIME"])
df = df.set_index(df.DATE)
# 1: spring,2: summer,3: fall,4: winter
df["season"]= df.index.month%12 // 3 + 1
#%%
# Citation: JAGANADH GOPINADHAN
# Check if holiday or not
cal = calendar()
holidays = cal.holidays(start = df.index.min(), end = df.index.max())
df["HOLIDAY"] = df.index.strftime('%Y-%m-%d').isin(holidays.strftime('%Y-%m-%d'))
df["HOLIDAY"] = pd.Categorical(df["HOLIDAY"])
df.season = pd.Categorical(df.season)
#%%
df.COST = df.COST.str[1:]
df.COST = df.COST.astype(float)
#%%

#%%
# Data Pre-processing 
display(df.head())
display(df.tail())
missing_vals = df.isna().sum()
print("The numeber of Missing Values:", missing_vals)


#%%
# Dropping column with empty value and irrelevant values
df = df.drop(columns=["NOTES","TYPE","UNITS"])


# Displaying the Shape of the dataframe
print(f"There are {len(df.index)} rows and {len(df.columns)} columns in the dataset.")
display(df.head())

#%%
# Citation : SANKET SHARMA
# Hourly
hourly_usage_df = df.resample("1H", on="DATE").sum()
display(hourly_usage_df.head())

# Daily
daily_usage_df = hourly_usage_df.resample("1D").sum()
display(daily_usage_df.head())

# Monthly
monthly_usage_df = daily_usage_df.resample("1M").sum()
display(monthly_usage_df.head())

#%%
monthly_usage_df["per_unit_cost"] = monthly_usage_df["COST"]/monthly_usage_df["USAGE"]
monthly_usage_df["year"] = monthly_usage_df.index.year.astype(str)
monthly_usage_df["month_name"] = monthly_usage_df.index.month_name().str[:3]

monthly_usage_df["usage_month"] = monthly_usage_df["month_name"] + "-" + monthly_usage_df["year"]

monthly_usage_df = monthly_usage_df.drop(columns=["year", "month_name"], errors="ignore")

hourly_usage_df["year"] = hourly_usage_df.index.year.astype(str)
hourly_usage_df["month_name"] = hourly_usage_df.index.month_name().str[:3]

hourly_usage_df["usage_month"] = hourly_usage_df["month_name"] + "-" + hourly_usage_df["year"]
hourly_usage_df["season"]= hourly_usage_df.index.month%12 // 3 + 1
hourly_usage_df.season = pd.Categorical(hourly_usage_df.season)

hourly_usage_df = hourly_usage_df.drop(columns=["year", "month_name"], errors="ignore")
hourly_usage_df["HOLIDAY"] = hourly_usage_df.index.strftime('%Y-%m-%d').isin(holidays.strftime('%Y-%m-%d')).astype(int)
hourly_usage_df["HOLIDAY"] = pd.Categorical(hourly_usage_df["HOLIDAY"])

display(hourly_usage_df.head())
#%%
tx.ACF_PACF_Plot(hourly_usage_df.USAGE,25)
#%%
# Target variable vs Time
plt.plot(df.USAGE)
plt.title("Usage in every 15 Minutes")
plt.xlabel("Month")
plt.ylabel("Electricity Usage in KWh")
plt.xticks(rotation = 90)
plt.show()
#%%
# Change in unit rate montly
plt.plot(monthly_usage_df.index,monthly_usage_df.per_unit_cost)
plt.title("Monthly Change in Price per unit")
plt.xlabel("Month")
plt.ylabel("Electricity price/unit ($)")
plt.xticks(rotation = 90)
plt.show()
#%%
# Hourly change in usage of electricity
plt.plot(hourly_usage_df.USAGE)
plt.title("Hourly Usage")
plt.xlabel("Month")
plt.ylabel("Electricity Usage in KWh")
plt.xticks(rotation = 90)
plt.show()
#%%

#%%
datatypes = hourly_usage_df.dtypes
display(datatypes)
#%%
X_train, X_test = train_test_split(hourly_usage_df,test_size=0.2,shuffle=False)
# %%
tx.Cal_rolling_mean_var(hourly_usage_df.USAGE)
# Since the rolling mean and the rolling variance turns into a straight line the data is stationary.
# %%
tx.ADF_Cal(hourly_usage_df.USAGE)
# We reject the null hypothesis hence the data is staitonary according to ADF test
# %%
tx.kpss_test(hourly_usage_df.USAGE)
# We failt reject the null hypothesis hence the data is staitonary according to kpss test
#%%
# %%
tx.stem_plot(hourly_usage_df.USAGE,20,name="Hourly electricity usage")
# %%
sns.pairplot(hourly_usage_df,kind="kde").set(title='Hourly Usage')
# %%
df1 = hourly_usage_df.corr()
sns.heatmap(df1,annot=True).set(title = "Correlation matrix for hourly Usage ")
# %%
STL = STL(hourly_usage_df.USAGE)
res = STL.fit()
fig = res.plot()
plt.show()
#%%
S=  res.seasonal
T =  res.trend
R = res.resid
#%%
plt.plot(R,label="Residual")
plt.plot(S,label="Seasonal")
plt.plot(T,label="Trend")
plt.legend(loc = "best")
plt.xlabel("months")
plt.ylabel("Electricity Usage in kWh")
plt.title("Plot for seasonality, trend and residual")
plt.xticks(rotation = 90)
plt.show()
# %%
adj_seasonal  = hourly_usage_df.USAGE-S
plt.plot(hourly_usage_df.USAGE,label="original")
plt.plot(adj_seasonal,label="seasonally adjusted")
plt.xlabel("Month")
plt.ylabel("Electricity Usage in kWh")
plt.title("Orginal data vs seasonally adjusted data")
plt.legend(loc = "best")
plt.xticks(rotation = 90)
plt.show
# %%
F = np.maximum(0,1-np.var(np.array(R))/np.var(np.array(S)+np.array(R)))
print(f"The strength of seasonality for this data set is:{F}")
F1 = np.maximum(0,1-np.var(np.array(R))/np.var(np.array(T)+np.array(R)))
print(f"The strength of trend for this data set is:{F1}")
#%%
detrended_pass= hourly_usage_df.USAGE-T
plt.plot(hourly_usage_df.USAGE,label="original")
plt.plot(detrended_pass,label="Detrended data")
plt.xlabel("Month")
plt.ylabel("Electricity Usage in kWh")
plt.title("Orginal data vs detrended data")
plt.legend(loc = "best")
plt.xticks(rotation = 90)
plt.show
# %%
holtt = ets.ExponentialSmoothing(X_train.USAGE,seasonal ="add",damped_trend = False).fit()
holtf = holtt.forecast(steps=len(X_test.USAGE))
holtf = pd.DataFrame(holtf).set_index(X_test.index)

#%%
holtt.summary()
#%%
error_hw = X_test.USAGE - holtf[0]
er_sqd_hw = error_hw**2
pred_error_hw = X_train.USAGE-holtt.fittedvalues

rk= tx.stem_plot(pred_error_hw,5,name="predcition errors")
qv = (len(X_train))*(np.array(rk[1:])**2)
af = print(f"Name of test:Holt's Winter Method,\n 'Q value':{qv},MSE for prediction errors: {np.mean((X_train.USAGE-holtt.fittedvalues)**2)},MSE for forecast errors:{np.mean(er_sqd_hw)},Variance of prediction errors:{np.var(X_train.USAGE-holtt.fittedvalues)},Variance of forecast errors: {np.var(error_hw)}, Correlation Coeff:{np.corrcoef(er_sqd_hw,X_test.USAGE)}")
# tab = tab.append(af,ignore_index = True)

fig, ax = plt.subplots()
ax.plot(X_train.USAGE,label = "Training set")
ax.plot(X_test.USAGE,label ="Test")
ax.plot(holtf[0],label = "Holt's Winter Method")
ax.set_xlabel("Months")
ax.set_ylabel("Electricity Usage in kWh")
ax.set_title("Holt's Winter Method")
plt.xticks(rotation = 90)
ax.legend()
plt.show()
# %%
# Average forecasting
tx.average_forecasting(hourly_usage_df.USAGE)
# %%
# Naive forecasting
tx.N_forecasting(hourly_usage_df.USAGE)
# %%
# SES forecasting
tx.ses_forecasting(hourly_usage_df.USAGE,alpha=0.1)
# %%
# Drift method forecasting
tx.d_forecasting(hourly_usage_df.USAGE)
# %%
X= hourly_usage_df.loc[:,hourly_usage_df.columns!='USAGE']
X= X.drop(columns="usage_month")
Y = hourly_usage_df.loc[:,hourly_usage_df.columns=='USAGE']
X = sm.add_constant(X)
#%%
X_tr , X_ts,Y_tr, Y_ts = train_test_split(X,Y,test_size=0.2, shuffle=False)
#%%
Xm = np.array(X_tr)
Ym = np.array(Y_tr)
h = np.matmul(Xm.T,Xm)
# %%
s, d, v = np.linalg.svd(h)
# %%
print("Singular Values =",d)
print("The conditional number =",np.linalg.cond(Xm))
# %%
# Coefficients from MLR
model = sm.OLS(Y_tr,X_tr).fit()
print(model.summary())
# %%
X_tr = X_tr.drop(columns="season")
# %%
model = sm.OLS(Y_tr,X_tr).fit()
print(model.summary())
Xm = np.array(X_tr)
Ym = np.array(Y_tr)
h = np.matmul(Xm.T,Xm)
s, d, v = np.linalg.svd(h)
print("Singular Values =",d)
print("The conditional number =",np.linalg.cond(Xm))
# %%
X_ts = X_ts.drop(columns="HOLIDAY")
prediction = model.predict(X_ts)
print(prediction)
# %%
plt.plot(Y_tr,label = "Train Set")
plt.plot(Y_ts,label = "Test Set")
plt.plot(prediction, label = "Forecast")
plt.legend()
plt.xlabel("Month")
plt.ylabel("Usage")
plt.title("Train, Test and the forecast after backward feature selection")
plt.xticks(rotation = 90)
plt.show()
# %%
pred_error = Y_tr.subtract(model.fittedvalues,axis=0)
print(pred_error)

pred_error = np.array(pred_error)
# %%
rk = tx.stem_plot(pred_error,lag = 50,name = "Prediction Error")
print("ACF of resioduals:",rk)
# %%
# var_pred = np.sqrt((1/(len(pred_error)-len(X_test.columns)))*((sum(i*i for i in pred_error))))
print("MSE for prediction errors:",np.mean(np.square(pred_error)))
print("The Variance for prediction error:",np.var(pred_error))

for_error = Y_ts.USAGE- prediction
print("The MSE for forecast errors:",np.mean(np.square(for_error)))
print("The variance for the forecast error:",np.var(for_error))

Q = len(X_train) * np.sum(np.square(rk[20:]))
sm.stats.acorr_ljungbox(for_error, lags=[20], return_df=True)
print("Q Value for MLR:",Q)


# %%
tx.arma_dat(hourly_usage_df.USAGE,na =7,nb = 7)
# %%
model = sm.tsa.ARIMA(hourly_usage_df.USAGE, order=(2, 0, 0), trend="n", ).fit()
print(model.summary())
# Prediction
model_hat = model.predict(start=0, end=len(X_train) - 1)
pred_error = X_train.USAGE[1:] - model_hat[:-1]
re = tx.stem_plot(pred_error, 20, name="Errors")
Q = len(X_train) * np.sum(np.square(re[20:]))
DOF = 20 - 2 - 0
alfa = 0.01
chi_critical = chi2.ppf(1 - alfa, DOF)
if Q < chi_critical:
    print("The residual is white")
else:
    print("The residual is NOT white")
lbvalue, pvalue = sm.stats.acorr_ljungbox(pred_error, lags=[20], return_df=True)
print(lbvalue)
print(pvalue)
plt.figure()
plt.plot(X_train.USAGE, "r", label="Train Data")
plt.plot(model_hat, "b", label="Prediction")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.legend()
plt.xticks(rotation = 90)
plt.title("statsmodel ARMA process generation, parameter estimation and prediction")
plt.show()
#%%
forecast = model.predict(start=len(X_train), end=len(hourly_usage_df))
for_error = X_test.USAGE-forecast
re = tx.stem_plot(for_error, 20, name="Errors")
Q = len(X_test) * np.sum(np.square(re[20:]))
DOF = 20 - 2 - 0
alfa = 0.01
chi_critical = chi2.ppf(1 - alfa, DOF)
if Q < chi_critical:
    print("The residual is white")
else:
    print("The residual is NOT white")
lbvalue, pvalue = sm.stats.acorr_ljungbox(pred_error, lags=[20], return_df=True)
print(lbvalue)
print(pvalue)
plt.figure()
plt.plot(X_test.USAGE, "r", label="Train Data")
plt.plot(forecast, "b", label="Forecast")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.legend()
plt.title("statsmodel ARMA process generation, parameter estimation and prediction")
plt.show()
#%%
print("MSE for prediction errors:",np.mean(np.square(pred_error)))
print("The Variance for prediction error:",np.var(pred_error))

print("MSE for forecast errors:",np.mean(np.square(for_error)))
print("The Variance for forecast error:",np.var(for_error))
#%%
diff1 = tx.difference(hourly_usage_df.USAGE,12)
tx.ACF_PACF_Plot(diff1,50)
# %%
# ARIMA(0,12,1)
model = sm.tsa.ARIMA(X_train.USAGE, order=(0, 12, 1) ).fit()
print(model.summary())
# Prediction
model_hat = model.predict(start=0, end=len(X_train))
e =  X_train.USAGE - model_hat

print("MSE for Prediction:",np.mean(np.square(e)))
print("Variance for Prediction:",np.var(e))
# forecast

forecast = model.predict(start = len(X_train),end  = len(hourly_usage_df))
e =  X_test.USAGE - forecast

print("MSE for forecast:",np.mean(np.square(e)))
print("Variance for Forecast:",np.var(e))

re = tx.stem_plot(e, 20, name="Errors")
Q = len(X_train.USAGE) * np.sum(np.square(re[20:]))
DOF = 20 - 0 - 1
alfa = 0.01
chi_critical = chi2.ppf(1 - alfa, DOF)
if Q < chi_critical:
    print("The residual is white")
else:
    print("The residual is NOT white")
lbvalue, pvalue = sm.stats.acorr_ljungbox(e, lags=[20], return_df=True)
print(lbvalue)
print(pvalue)
plt.figure()
plt.plot(X_train.USAGE, "r", label="True Data")
plt.plot(model_hat, "b", label="Fitted Data")
plt.xlabel("Samples")
plt.ylabel("Electricity Usage in kWh")
plt.legend()
plt.xticks(rotation = 90)
plt.title("statsmodel ARIMA process")
plt.show()
#%%
plt.plot(X_test.USAGE,label = "Test Data")
plt.plot(forecast,label = "Forecast")
plt.legend()
plt.xlabel("Months")
plt.ylabel("Electricity Usage in kWh")
plt.title("statsmodel ARIMA process")
plt.xticks(rotation = 90)
plt.show()
#%%
# SARIMA(0,1,1)
model=sm.tsa.statespace.SARIMAX(X_train.USAGE,order=(0, 0, 0),seasonal_order=(0,1,1,12))
results=model.fit()

print(results.summary())

prediction = results.predict(start = 0,end =len(X_train.USAGE),dynamic = False)
plt.plot(X_train.USAGE,label = "Train Data")
plt.plot(prediction,label ="Prediction")
plt.legend()
plt.xlabel("Month")
plt.ylabel("Electricity Usage in kWh")
plt.title("SARIMA train vs prediction")
plt.xticks(rotation = 90)
#%%
prediction_error = X_train.USAGE-prediction
print("MSE for SARIMA model (prediction):",np.mean(np.square(prediction_error)))
print("Variance for SARIMA model (prediction):",np.var(prediction_error))
# %%
forecast = results.predict(start = len(X_train),end =len(hourly_usage_df.USAGE),dynamic = False)
plt.plot(X_test.USAGE,label = "Test Data")
plt.plot(forecast,label ="Prediction")
plt.legend()
plt.xlabel("Month")
plt.ylabel("Electricity Usage in kWh")
plt.title("SARIMA train vs prediction")
plt.xticks(rotation = 90)
# %%
forecast_error = X_test.USAGE-forecast
print("MSE for SARIMA model (forecast):",np.mean(np.square(forecast_error)))
print("Variance for SARIMA model (forecast):",np.var(forecast_error))
# %%
re = tx.stem_plot(forecast_error, 20, name="Errors")
Q = len(X_train.USAGE) * np.sum(np.square(re[20:]))
DOF = 20 - 0 - 1
alfa = 0.01
chi_critical = chi2.ppf(1 - alfa, DOF)
if Q < chi_critical:
    print("The residual is white")
else:
    print("The residual is NOT white")
lbvalue, pvalue = sm.stats.acorr_ljungbox(e, lags=[20], return_df=True)
print(lbvalue)
print(pvalue)
# %%
