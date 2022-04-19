#%%
from multiprocessing.spawn import import_main_path
from unicodedata import category
from importlib_metadata import distribution
import pandas as pd
# from Lab4 import X_test, X_train
import numpy as np
import toolbox as tx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import math
# %%
def unique_value_count(df: pd.DataFrame):
    df_nunique = df.nunique().reset_index()
    df_nunique['dtype'] = df.dtypes.reset_index().loc[:, 0]
    df_nunique.columns = ['column', 'nunique', 'dtype']
    
    display(df_nunique)

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
df  = pd.read_csv("/Users/atharvah/GWU/Untitled/personal/Time_Series/Labs/D202.csv")
df.DATE = pd.to_datetime(df.DATE + ' ' + df["END TIME"])
df = df.drop(columns=["START TIME","END TIME"])
df = df.set_index(df.DATE)
df.COST = df.COST.str[1:]
df.COST = df.COST.astype(float)
#%%
plt.plot(df.USAGE)
plt.title("Usage in every 15")
plt.xlabel("Month")
plt.ylabel("Electricity Usage in KWh")
plt.xticks(rotation = 90)
plt.show()
#%%
# Data Pre-processing 
display(df.head())
display(df.tail())
missing_vals = df.isna().sum()
print("The numeber of Missing Values:", missing_vals)


#%%
# Check if holiday or not
cal = calendar()
holidays = cal.holidays(start = df.index.min(), end = df.index.max())


# Dropping column with empty value and irrelevant values
unique_value_count(df=df)
df = df.drop(columns=["NOTES","TYPE","UNITS"])


# Displaying the Shape of the dataframe
print(f"There are {len(df.index)} rows and {len(df.columns)} columns in the dataset.")
display(df.head())


#%%
# Check if holiday or not
cal = calendar()
holidays = cal.holidays(start = df.index.min(), end = df.index.max())
df["HOLIDAY"] = df.index.strftime('%Y-%m-%d').isin(holidays.strftime('%Y-%m-%d'))
df["HOLIDAY"] = pd.Categorical(df["HOLIDAY"])
#%%
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

daily_usage_df["per_unit_cost"] = daily_usage_df["COST"]/daily_usage_df["USAGE"]
daily_usage_df["year"] = daily_usage_df.index.year.astype(str)
daily_usage_df["month_name"] = daily_usage_df.index.month_name().str[:3]

daily_usage_df["usage_month"] = daily_usage_df["month_name"] + "-" + daily_usage_df["year"]

daily_usage_df = daily_usage_df.drop(columns=["year", "month_name"], errors="ignore")

hourly_usage_df["per_unit_cost"] = hourly_usage_df["COST"]/hourly_usage_df["USAGE"]
hourly_usage_df["year"] = hourly_usage_df.index.year.astype(str)
hourly_usage_df["month_name"] = hourly_usage_df.index.month_name().str[:3]

hourly_usage_df["usage_month"] = hourly_usage_df["month_name"] + "-" + hourly_usage_df["year"]

daily_usage_df = daily_usage_df.drop(columns=["year", "month_name"], errors="ignore")
daily_usage_df["HOLIDAY"] = daily_usage_df.index.strftime('%Y-%m-%d').isin(holidays.strftime('%Y-%m-%d'))
daily_usage_df["HOLIDAY"] = pd.Categorical(daily_usage_df["HOLIDAY"])

hourly_usage_df = hourly_usage_df.drop(columns=["year", "month_name"], errors="ignore")
hourly_usage_df["HOLIDAY"] = hourly_usage_df.index.strftime('%Y-%m-%d').isin(holidays.strftime('%Y-%m-%d')).astype(int)
hourly_usage_df["HOLIDAY"] = pd.Categorical(hourly_usage_df["HOLIDAY"])

display(daily_usage_df.head())
display(monthly_usage_df.head())
display(hourly_usage_df.head())
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
tx.stem_plot(hourly_usage_df.USAGE,50)
#%%
diff1_df = difference(hourly_usage_df.USAGE,15)
#%%
diff2_df = difference(diff1_df,15)
#%%
diff3_df = difference(diff2_df,15)
#%%
log = log_trans(hourly_usage_df.USAGE)
diff1_df = difference(log,15)
#%%
datatypes = hourly_usage_df.dtypes
display(datatypes)
#%%
X_train, X_test = train_test_split(hourly_usage_df,test_size=0.2)
# %%
tx.Cal_rolling_mean_var(hourly_usage_df.USAGE)
# Since the rolling mean and the rolling variance turns into a straight line the data is stationary.
# %%
tx.ADF_Cal(hourly_usage_df.USAGE)
# We reject the null hypothesis hence the data is staitonary according to ADF test
# %%
tx.kpss_test(hourly_usage_df.USAGE)
# We failt reject the null hypothesis hence the data is staitonary according to kpss test
# %%
tx.stem_plot(hourly_usage_df.USAGE,20,name="Hourly electricity usage")
# %%
sns.pairplot(hourly_usage_df,kind="kde").set(title='Hourly Usage')
sns.pairplot(monthly_usage_df,kind="kde").set(title='Monthly Usage')
sns.pairplot(daily_usage_df,kind="kde").set(title='Daily Usage')

# %%
df1 = hourly_usage_df.corr()
sns.heatmap(df1,annot=True).set(title = "Correlation matrix for hourly Usage ")
# %%
tx.N_forecasting(hourly_usage_df.COST,table=True,q_val=True)
# %%
tx.d_forecasting(hourly_usage_df.COST,table=True,q_val=True)
# %%
tx.ses_forecasting(hourly_usage_df.COST,table=True,q_val=True,alpha=0.5)
# %%
tx.ma(hourly_usage_df.COST,3)
# %%
