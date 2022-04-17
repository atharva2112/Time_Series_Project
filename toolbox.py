#%%
from cProfile import label
import math
from random import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import mean, number, product, square
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statistics
from tabulate import tabulate
from sklearn.model_selection import train_test_split
#%%
def difference(dataset, interval =1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i]- dataset[i-interval]
        diff.append(value)
    return diff
# Find correlation coefficient
def correlation_coefficent_cal(x,y):
    numerator = sum((x-np.mean(x))*(y-np.mean(y)))
    denominator = math.sqrt(sum((x-np.mean(x))**2))*math.sqrt(sum((y-np.mean(y))**2))
    cor =round((numerator/denominator),2)
    if type(x)==list:
        print(f"The correlation coefficient for x and y: ",round((numerator/denominator),2))
    else:
        print(f"The correlation coefficient for {x.name} and {y.name}: ",round((numerator/denominator),2))
    return cor
#%%
# Scatter plot for two columns to analyse correlation
def plotting(x,y):
   sns.scatterplot(x,y)
   plt.title(f"Scatter plot for {x.name} and {y.name}")
   plt.show()
#%%
# To find mean,variance and standard deviation for a column/list
def series_stats(data):
    i=1
    while i < len(data.columns):
        print("The %s mean is : %f and the variance is : %f with standard deviation :%f"%(data.columns[i],statistics.mean(data.loc[:,data.columns[i]]),statistics.variance(data.loc[:,data.columns[i]]),statistics.stdev(data.loc[:,data.columns[i]])))
        i+=1
#%%
# Calculate rolling mean and rolling variance to know if data is stationary or not.
def Cal_rolling_mean_var(column):
    i = 1
    mean_list = []
    var_list = []
    if type(column) == list:
        column= pd.Series(column)
    while i < len(column):
        mean_list.append(statistics.mean(column.head(i)))
        i += 1
        var_list.append(statistics.variance(column.head(i)))
    plt.subplot(3, 1, 1)
    plt.plot(mean_list)
    plt.title("Rolling Mean Graph")
    plt.subplot(3, 1, 3)
    plt.plot(var_list)
    plt.title("Rolling Variance Graph")
    plt.show()
#%%
# ADF test
def ADF_Cal(x):
 result = adfuller(x)
 print("ADF Statistic: %f" %result[0])
 print('p-value: %f' % result[1])
 print('Critical Values:')
 for key, value in result[4].items():
     print('\t%s: %.3f' % (key, value))
#%%
# KPSS test
def kpss_test(timeseries):
 print ('Results of KPSS Test:')
 kpsstest = kpss(timeseries, regression='c', nlags="auto")
 kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','LagsUsed'])
 for key,value in kpsstest[3].items():
     kpss_output['Critical Value (%s)'%key] = value
     print (kpss_output)
#%%
# Calculate acf for N lags 
def acf(df, lag):
    numerator = 0
    denominator = 0
    global auto
    auto = []
    avg = np.mean(df)
    for i in range(lag,len(df)):
        numerator += (df[i] - avg)*(df[i-lag] - avg)
    for i in range(len(df)):
        denominator += (df[i]-avg)**2
    auto=(numerator/denominator)
    return auto
#%%
# Plot stem plot for N lags
def stem_plot(df,lag,ax=None,markerfmt ="o",name = None):
    a = []
    b = []
    i=0
    ax= ax or plt.gca()
    for i in range(lag+1):
        a.append(acf(df,i))
        b.append(i)
    ax.stem(b,a)
    ax.stem(np.negative(b),a)
    ax.axhspan(-(1.96/math.sqrt(len(df))),(1.96/math.sqrt(len(df))), color='blue', alpha=0.2)
    ax.set_xlabel("Lags")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"ACF plot for {name}")
    ax.set_xlim(-lag,lag)
    plt.show
    return a
#%%
# Average Method Forecasting
def average_forecasting(tr_data,h=None,q_val = False):
    forecast = [0]
    error = [0]
    er_sqd = [0]
    value = 0
    X_train, X_test = train_test_split(tr_data,test_size=0.2,shuffle=False)
    if h==None:
        for i in range(len(X_train)):
            if i>0:
                f = X_train[0:i]
                value = mean(f)
                forecast.append(value)
                value1 = X_train[i]-value
                error.append(value1)
                er_sqd = [i ** 2 for i in error]
    elif (h!=1):
        for i in range(len(X_train)):
            f = X_train[0:i+h]
            value = mean(f)
            forecast.append(value)
            value1 = X_train[i]-value
            error.append(value1)
            er_sqd = [i ** 2 for i in error]
    print("For train set")
    info={"Y(t)":X_train,"Forecast":forecast,"Errors":error,"Squared Errors":er_sqd}
    print(tabulate(info,headers="keys",tablefmt="fancy_grid",showindex=True))
    print("The mean squared error(MSE):",mean(er_sqd),"\n")
    
    print("The variance for prediction errors:",np.var(error[:len(X_train)]))
     
    for i in range(len(X_test)):
        forecast.append(forecast[-1])
        value1 = X_test[i]-value
        error.append(value1)
        er_sqd = [i ** 2 for i in error]
    print("For test set")
    info={"Y(t)":X_test,"Forecast":forecast[len(X_train):],"Errors":error[len(X_train):],"Squared Errors":er_sqd[len(X_train):]}
    print(tabulate(info,headers="keys",tablefmt="fancy_grid",showindex=True))
    print("The mean squared error(MSE):",mean(er_sqd))
    
    print("The Variance for the forecast error:",np.var(error[len(X_train):]))
    
    fig, ax= plt.subplots()
    ax.plot(X_test,label="Test Set")
    ax.plot(X_train,label="Train Set")
    ax.plot(forecast[len(X_train):],label="H-Step Forecast")
    ax.legend()
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Index")
    ax.set_title("Average Method Forecasting")
    plt.show()
    
    if q_val==True:
        rk= stem_plot(error[:len(X_train)],5)
        qv = (len(tr_data))*(rk**2)
        print("The Q-value for the training set:",qv)
        return qv
    return forecast,error,er_sqd
#%%
# %%
def N_forecasting(tr_data,h=None,q_val = False):
    forecast = [0]
    error = [0]
    er_sqd = [0]
    value = 0
    X_train, X_test = train_test_split(tr_data,test_size=0.2,shuffle=False)
    if h==None:
        for i in range(len(X_train)):
            if i>0:
                forecast.append(X_train[i-1])
                value1 = X_train[i]-value
                error.append(value1)
                er_sqd = [i ** 2 for i in error]
    print("For train set")
    info={"Y(t)":X_train,"Forecast":forecast,"Errors":error,"Squared Errors":er_sqd}
    print(tabulate(info,headers="keys",tablefmt="fancy_grid",showindex=range(1,len(forecast)+1)))
    print("The mean squared error(MSE):",mean(er_sqd),"\n")
    print("The variance for prediction errors:",np.var(error[:len(X_train)]))
     
    for i in range(len(X_test)):
        forecast.append(X_train[-1])
        value1 = X_test[i]-value
        error.append(value1)
        er_sqd = [i ** 2 for i in error]
    print("For test set")
    info={"Y(t)":X_test,"Forecast":forecast[len(X_train):],"Errors":error[len(X_train):],"Squared Errors":er_sqd[len(X_train):]}
    print(tabulate(info,headers="keys",tablefmt="fancy_grid",showindex=True))
    print("The mean squared error(MSE):",mean(er_sqd))
    
    print("The Variance for the forecast error:",np.var(error[len(X_train):]))
    
    fig, ax= plt.subplots()
    ax.plot(X_test,label="Test Set")
    ax.plot(X_train,label="Train Set")
    ax.plot(forecast[len(X_train):],label="H-Step Forecast")
    ax.legend()
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Index")
    ax.set_title("Naive Method Forecasting")
    plt.show()
    
    if q_val==True:
        rk= acf(error[:len(X_train)],5)
        qv = (len(tr_data))*(rk**2)
        print("The Q-value for the training set:",qv)
        return qv
    return forecast,error,er_sqd
# %%
def d_forecasting(tr_data,h=None,q_val = False):
    forecast = [0]
    error = [0]
    er_sqd = [0]
    value = 0
    X_train, X_test = train_test_split(tr_data,test_size=0.2,shuffle=False)
    if h==None:
        for i in range(len(X_train)):
            if i>0:
                f = X_train[0:i+1]
                value = X_train[i-1] + ((i+1)*((X_train[i-1]-X_train[0])/(len(tr_data)-1)))
                forecast.append(value)
                value1 = X_train[i]-value
                error.append(value1)
                er_sqd = [i ** 2 for i in error]
    print("For train set")
    info={"Y(t)":X_train,"Forecast":forecast,"Errors":error,"Squared Errors":er_sqd}
    print(tabulate(info,headers="keys",tablefmt="fancy_grid",showindex=range(1,len(forecast)+1)))
    print("The mean squared error(MSE):",mean(er_sqd),"\n")
    
    print("The variance for prediction errors:",np.var(error[:len(X_train)]))
     
    for i in range(len(X_test)):
        forecast.append(forecast[-1])
        value1 = X_test[i]-value
        error.append(value1)
        er_sqd = [i ** 2 for i in error]
    print("For test set")
    info={"Y(t)":X_test,"Forecast":forecast[len(X_train):],"Errors":error[len(X_train):],"Squared Errors":er_sqd[len(X_train):]}
    print(tabulate(info,headers="keys",tablefmt="fancy_grid",showindex=True))
    print("The mean squared error(MSE):",mean(er_sqd))
    
    print("The Variance for the forecast error:",np.var(error[len(X_train):]))
    
    fig, ax= plt.subplots()
    ax.plot(X_test,label="Test Set")
    ax.plot(X_train,label="Train Set")
    ax.plot(forecast[len(X_train):],label="H-Step Forecast")
    ax.legend()
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Index")
    ax.set_title("Average Method Forecasting")
    plt.show()
    
    if q_val==True:
        rk= acf(error[:len(X_train)],5)
        qv = (len(tr_data))*(rk**2)
        print("The Q-value for the training set:",qv)
        return qv
    return forecast,error,er_sqd
#%%
def ses_forecasting(tr_data,alpha,h=None,q_val = False,plot = None,ax=None):
    
    
    forecast1 = [0]
    error = [0]
    er_sqd = [0]
    value = 0
    X_train, X_test = train_test_split(tr_data,test_size=0.2,shuffle=False)
    forecast1 = [X_train[0]]
    
    for i in range(len(X_train)):
        
        f = X_train[0:i+1]
        value = (X_train[i-1]*alpha)+((1-alpha)*forecast1[i-1])
        forecast1.append(value)
        value1 = X_train[i]-value
        error.append(value1)
        er_sqd = [i ** 2 for i in error]
    if plot==False or plot== None:
        print("For train set")
        info={"Y(t)":X_train,"Forecast":forecast1,"Errors":error,"Squared Errors":er_sqd}
        print(tabulate(info,headers="keys",tablefmt="fancy_grid",showindex=True))
        print("The mean squared error(MSE):",mean(er_sqd),"\n")

        print("The variance for prediction errors:",np.var(error[:len(X_train)]))
     
    for i in range(len(X_test)):
        forecast1.append(forecast1[-1])
        value1 = X_test[i]-value
        error.append(value1)
        er_sqd = [i ** 2 for i in error]
        
    if plot==False or plot== None:    
        print("For test set")
        info={"Y(t)":X_test,"Forecast":forecast1[len(X_train):],"Errors":error[len(X_train):],"Squared Errors":er_sqd[len(X_train):]}
        print(tabulate(info,headers="keys",tablefmt="fancy_grid",showindex=True))
        print("The mean squared error(MSE):",mean(er_sqd))

        print("The Variance for the forecast error:",np.var(error[len(X_train):]))
        # c = [(idx, item) for idx,item in enumerate(forecast, start=1)]
    if plot == True or plot== None:
        ax = ax or plt.gca()
        ax.plot(X_test,label="Test Set")
        plt.show
        ax.plot(X_train,label="Train Set")
        plt.show
        ax.plot(forecast1[len(X_train):],label="H-Step Forecast")
        plt.show
        ax.legend()
        ax.set_ylabel("Magnitude")
        ax.set_xlabel("Index")
        ax.set_title("Average Method Forecasting")
    

    if q_val==True:
        rk= acf(error[:len(X_train)],5)
        qv = (len(tr_data))*(rk**2)
        print("The Q-value for the training set:",qv)
        return qv
    tab = pd.DataFrame()
    return 
#%%
# ACF/PACF plot
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()
#%%
# gpac
def arma(na,nb):
    N= int(input("Enter the Number of Samples:"))
    mean_e  = int(input("Enter the mean for the WN"))
    var_e  = int(input("Enter the Variance for the WN"))
    e = np.random.normal(mean_e,var_e,N)
    y  = np.zeros(len(e))
    oa = int(input("Enter AR order"))
    ob = int(input("Enter MA order"))
    arparams = [1]
    maparams = [1]
    if oa>=1:
        for i in range(oa):
            ar = float(input(f"Enter the parameter a{i+1}"))
            arparams.append(ar)
    if ob>=1:    
        for i in range(ob):
            ma = float(input(f"Enter the parameter b{i+1}"))
            maparams.append(ma)
    ar = np.r_[arparams]
    ma = np.r_[maparams]
    arma_process = sm.tsa.ArmaProcess(ar,ma)
    print("Is this a stationary process:",arma_process.isstationary)
    y = arma_process.generate_sample(N)
    lags = int(input("Enter the number of lags:"))
    ry = arma_process.acf(lags=lags)
    print(ry.shape)
    matrix = np.matrix([[],[]])
    # for r in range(na):
    #     i = 1
    #     den [j-k+i]
    gpac = np.zeros(shape=(nb,na))
    gpac_den = np.zeros(shape=(na,na))
    gpac_num = np.zeros(shape=(na,na))
    for k in range(1,na+1):
        for j in range(nb):
            #denominator
            den = np.zeros(shape=(k,k))
            i = 1
            for c in range(k):
                for r in range(k):
                    den[r,c] = ry[abs(r-c+j)]
                gpac_den = np.linalg.det(den)
            #numerator
            num = np.zeros(shape=(k,k))
            for c in range(k):
                for r in range(k):
                    if c < k-1:
                        num[r,c] = ry[abs(r-c+j)]
                    else:
                        num[r,c] = ry[abs(j+r+1)]
                gpac_num = np.linalg.det(num)
            gpac[j,k-1] = round(gpac_num/gpac_den,2)
    print(gpac)
    sns.heatmap(gpac,annot=True,xticklabels=range(1,na+1))
    plt.ylabel("nb")
    plt.title("GPAC Table\nna")
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = True, labeltop=True)
    ACF_PACF_Plot(y,lags = 20)
#%%
def cat_var_checker(df, dtype='object'):
    # Get the dataframe of categorical variables and their number of unique value
    df_cat = pd.DataFrame([[var, df[var].nunique(dropna=False)]
                           # If the data type is dtype
                           for var in df.columns if df[var].dtype == dtype],
                          columns=['var', 'nunique'])
    
    # Sort df_cat in accending order of the number of unique value
    df_cat = df_cat.sort_values(by='nunique', ascending=False).reset_index(drop=True)
    
    return df_cat
# %%
def common_var_checker(df_train, df_val, df_test, target):
    # Get the dataframe of common variables between the training, validation and test data
    df_common_var = pd.DataFrame(np.intersect1d(np.intersect1d(df_train.columns, df_val.columns), np.union1d(df_test.columns, [target])),
                                 columns=['common var'])
                
    return df_common_var
# %%
