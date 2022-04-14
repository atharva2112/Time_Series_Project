#%%
from multiprocessing.spawn import import_main_path
import pandas as pd
from Lab4 import X_test, X_train
import numpy as np
import toolbox as tx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
# %%
df  = pd.read_csv("/Users/atharvah/GWU/Untitled/personal/Time_Series/Labs/D202.csv")
df.DATE = pd.to_datetime(df.DATE + ' ' + df["END TIME"])
df = df.drop(columns=["START TIME","END TIME"])
df = df.set_index(df.DATE)
df.COST = df.COST.str[1:]
df.COST = df.COST.astype(float)
#%%
# Data Pre-processing 
display(df.head())
display(df.tail())
missing_vals = df.isna().sum()
print("The numeber of Missing Values:", missing_vals)
#%%
# Dropping column with empty value
df = df.drop(columns=["NOTES"])
# Displaying the Shape of the dataframe
print(f"There are {len(df.index)} rows and {len(df.columns)} columns in the dataset.")
display(df.head())
#%%
df_common_var = tx.common_var_checker(df)
# Print df_common_var
df_common_var
#%%
plt.plot(df.USAGE)
plt.title("Electricity Usage everyday every 15 minutes")
plt.xticks(rotation = 90)
plt.xlabel("Months")
plt.ylabel("Electricity Usage in KWh")
datatypes = df.dtypes
display(datatypes)

#%%
X_train, X_test = train_test_split(df)
# %%
tx.Cal_rolling_mean_var(df.USAGE)
# %%
tx.ADF_Cal(df.USAGE)
# %%
tx.kpss_test(df.USAGE)
# %%
tx.stem_plot(df.USAGE,30)
# %%
