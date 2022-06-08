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
import statsmodels.api as sm
from scipy.stats import chi2
from scipy import signal
import tabulate
#%%
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

def average_forecasting(tr_data, h=None, q_val=False, table=False, index=None):
    tab = pd.DataFrame(columns=["Name of test","Q value","MSE for prediction errors","MSE for forecast errors","Mean of prediction errors","Variance of prediction errors","Variance of forecast errors"])
    global pred_error_avg
    global for_error_avg
    forecast = [None]
    error = [None]
    er_sqd = []
    error1 = []
    value = 0
    forecast1 = []
    X_train, X_test = train_test_split(tr_data, test_size=0.2, shuffle=False)
    if h == None:
        for i in range(len(X_train)):
            if i > 0:
                f = X_train[0:i]
                value = np.mean(f)
                forecast.append(value)
                value1 = X_train[i] - value
                error.append(value1)
                er_sqd = [i ** 2 for i in error if i != None]
            er_sqd.insert(0, None)
    print("For train set")
    info = {"Y(t)": X_train, "Forecast": forecast, "Errors": error, "Squared Errors": er_sqd}
    print(tabulate.tabulate(info, headers="keys", tablefmt="fancy_grid"))
    print("The mean squared error(MSE):", np.mean(er_sqd[1:]))

    print("The variance for prediction errors:", np.var(error[1:len(X_train)]))
    # error = pd.DataFrame(error).set_index(X_train.Passengers.index)
    for i in range(len(X_test)):
        if i == 0:
            f = X_train[:X_test.first_valid_index()]
            value = np.mean(f)
            forecast1.append(value)
            value1 = X_test[X_test.first_valid_index()] - value
            error1.append(value1)

        else:
            forecast1.append(forecast1[-1])
            value1 = X_test[i] - forecast1[-1]
            error1.append(value1)

        # forecast1.append(forecast[-1])
        # value1 = X_test.Passengers[X_test.first_valid_index()]-forecast[-1]
        # error.append(value1)
        er_sqd1 = [i ** 2 for i in error1]
    h_step = pd.DataFrame(forecast1, columns=['y_hat']).set_index(X_test.index)
    print("For test set")
    info = {"Y(t)": X_test, "Forecast": forecast1, "Errors": error1, "Squared Errors": er_sqd1}
    print(tabulate.tabulate(info, headers="keys", tablefmt="fancy_grid"))
    print("The mean squared error(MSE):", np.mean(er_sqd1))

    print("The Variance for the forecast error:", np.var(error1))

    fig, ax = plt.subplots()
    ax.plot(X_test, label="Test Set")
    ax.plot(X_train, label="Train Set")
    ax.plot(h_step.y_hat, label="H-Step Forecast")
    ax.legend()
    ax.set_ylabel("Samples")
    ax.set_xlabel("Month")
    ax.set_title("Average Method Forecasting")
    plt.xticks(rotation = 90)
    plt.show()

    if q_val == True:
        rk = stem_plot(error[1:], 5)
        qv = len(X_train + X_test) * np.sum(np.square(rk[1:]))
        print("The Q-value for the training set:", qv)
    else:
        qv = " "
    pred_error_avg = error[1:len(X_train)]
    for_error_avg = error1
    if table == True:
        er = np.mean(pred_error_avg)
        vr = np.var(pred_error_avg)
        vfr = np.var(for_error_avg)
        af = {"Name of test": "Average", 'Q value': qv, 'MSE for prediction errors': np.mean(er_sqd[1:len(X_train)]),
              "MSE for forecast errors": np.mean(er_sqd1), "Mean of prediction errors": er,
              "Variance of prediction errors": vr, "Variance of forecast errors": vfr,
              "Correlation Coeff": correlation_coefficent_cal(for_error_avg, X_test)}
        tab = tab.append(af, ignore_index=True)
        display(tab)
    return
#%%
# %%
def N_forecasting(tr_data,h=None,q_val = False,table = False,index = None):
    tab = pd.DataFrame(columns=["Name of test","Q value","MSE for prediction errors","MSE for forecast errors","Mean of prediction errors","Variance of prediction errors","Variance of forecast errors"])
    global pred_error_n
    global for_error_n
    forecast = [None]
    error = [None]
    er_sqd = []
    error1=[]
    value = 0
    forecast1=[]
    X_train, X_test = train_test_split(tr_data,test_size=0.2,shuffle=False)
    if h==None:
        for i in range(len(X_train)):
            if i>0:
                value = X_train[i-1]
                forecast.append(X_train[i-1])
                value1 = X_train[i]-value
                error.append(value1)
                er_sqd = [i ** 2 for i in error if i!= None]
            er_sqd.insert(0,None)
    print("For train set")
    info={"Y(t)":X_train,"Forecast":forecast,"Errors":error,"Squared Errors":er_sqd}
    print(tabulate.tabulate(info,headers="keys",tablefmt="fancy_grid"))
    print("The mean squared error(MSE):",np.mean(er_sqd[1:]))
    
    print("The variance for prediction errors:",np.var(error[1:]))
     
    for i in range(len(X_test)):
        forecast1.append(X_train[-1])
        value1 = X_test[i]-forecast[-1]
        error1.append(value1)
        er_sqd1 = [i ** 2 for i in error1]
    h_step = pd.DataFrame(forecast1,columns=['y_hat']).set_index(X_test.index)
    print("For test set")
    info={"Y(t)":X_test,"Forecast":forecast1,"Errors":error1,"Squared Errors":er_sqd1}
    print(tabulate.tabulate(info,headers="keys",tablefmt="fancy_grid"))
    print("The mean squared error(MSE):",np.mean(er_sqd1))
    
    print("The Variance for the forecast error:",np.var(error1))
    
    fig, ax= plt.subplots()
    ax.plot(X_test,label="Test Set")
    ax.plot(X_train,label="Train Set")
    ax.plot(h_step.y_hat,label="H-Step Forecast")
    ax.legend()
    ax.set_ylabel("Samples")
    ax.set_xlabel("Month")
    ax.set_title("Naive Method Forecasting")
    plt.xticks(rotation = 90)
    plt.show()
    
    if q_val==True:
        rk= stem_plot(error[1:],5)
        qv = len(X_train+X_test)*np.sum(np.square(rk[1:]))
        print("The Q-value for the training set:",qv)
    else:
        qv = " "
    pred_error_n = error[1:]
    for_error_n = error1
    if table == True:    
        er = np.mean(pred_error_n)
        vr= np.var(pred_error_n)
        vfr = np.var(for_error_n)
        af = {"Name of test": "Naive", 'Q value':qv,'MSE for prediction errors': np.mean(er_sqd[1:len(X_train)]),"MSE for forecast errors":np.mean(er_sqd1),"Mean of prediction errors":er,"Variance of prediction errors": vr,"Variance of forecast errors": vfr,"Correlation Coeff":correlation_coefficent_cal(for_error_n,X_test)}
        tab = tab.append(af,ignore_index = True)
    display(tab)
    return 
# %%
def d_forecasting(tr_data,h=None,q_val = False,table = False,index = None):
    tab = pd.DataFrame(columns=["Name of test","Q value","MSE for prediction errors","MSE for forecast errors","Mean of prediction errors","Variance of prediction errors","Variance of forecast errors"])
    global pred_error_d
    global for_error_d
    forecast = [None,None]
    error = [None,None]
    er_sqd = [None,None]
    error1 = []
    value = 0
    forecast1=[]
    X_train, X_test = train_test_split(tr_data,test_size=0.2,shuffle=False)
    if h==None:
        for i in range(len(X_train)):
            if i>1:
                value = X_train[i-1] + ((1)*((X_train[i-1]-X_train[0])/(len(X_train[:(i-1)]-1))))
                forecast.append(value)
                value1 = X_train[i]-value
                error.append(value1)
                er_sqd = [i ** 2 for i in error if i!=None]
        er_sqd.insert(0,None)
        er_sqd.insert(1,None)
    print("For train set")
    info={"Y(t)":X_train,"Forecast":forecast,"Errors":error,"Squared Errors":er_sqd}
    print(tabulate.tabulate(info,headers="keys",tablefmt="fancy_grid"))
    print("The mean squared error(MSE):",np.mean(er_sqd[2:]))
    
    print("The variance for prediction errors:",np.var(error[2:]))
     
    for i in range(len(X_test)):
        value = X_train[len(X_train)-1] + ((i+1)*((X_train[len(X_train)-1]-X_train[0])/(len(X_train)-1)))
        forecast1.append(value)
        value1 = X_test[i]-forecast[-1]
        error1.append(value1)
        er_sqd1 = [i ** 2 for i in error1]
    h_step = pd.DataFrame(forecast1,columns=['y_hat']).set_index(X_test.index)
    print("For test set")
    info={"Y(t)":X_test,"Forecast":forecast1,"Errors":error1,"Squared Errors":er_sqd1}
    print(tabulate.tabulate(info,headers="keys",tablefmt="fancy_grid"))
    print("The mean squared error(MSE):",np.mean(er_sqd1))
    
    print("The Variance for the forecast error:",np.var(error1))
    
    fig, ax= plt.subplots()
    ax.plot(X_test,label="Test Set")
    ax.plot(X_train,label="Train Set")
    ax.plot(h_step.y_hat,label="H-Step Forecast")
    ax.legend()
    ax.set_ylabel("Samples")
    ax.set_xlabel("Month")
    ax.set_title("Drift Method Forecasting")
    plt.xticks(rotation = 90)
    plt.show()
    
    if q_val==True:
        rk= stem_plot(error[2:],5)
        qv = len(X_train+X_test)*np.sum(np.square(rk[1:]))
        print("The Q-value for the training set:",qv)
    else:
        qv = " "
    pred_error_d = error[2:]
    for_error_d = error1
    if table == True:    
        er = np.mean(pred_error_d)
        vr= np.var(pred_error_d)
        vfr = np.var(for_error_d)
        af = {"Name of test": "Drift", 'Q value':qv,'MSE for prediction errors': np.mean(er_sqd[2:]),"MSE for forecast errors":np.mean(er_sqd1),"Mean of prediction errors":er,"Variance of prediction errors": vr,"Variance of forecast errors": vfr,"Correlation Coeff":correlation_coefficent_cal(for_error_d,X_test)}
        tab = tab.append(af,ignore_index = True)
        display(tab)
    return 
#%%
def ses_forecasting(tr_data,alpha,h=None,q_val = False,table = False,index = None):
    global pred_error_s
    tab = pd.DataFrame(columns=["Name of test","Q value","MSE for prediction errors","MSE for forecast errors","Mean of prediction errors","Variance of prediction errors","Variance of forecast errors"])
    global for_error_s
    # forecast = [0]
    error = [None]
    error1=[]
    er_sqd = [None]
    value = 0
    X_train, X_test = train_test_split(tr_data,test_size=0.2,shuffle=False)
    forecast1 = [X_train[0]]
    forecast2=[]
    if h==None:
        for i in range(len(X_train)):
            if i>0:
                value = (X_train[i-1]*alpha)+((1-alpha)*forecast1[i-1])
                forecast1.append(value)
                value1 = X_train[i]-value
                error.append(value1)
                er_sqd = [i ** 2 for i in error if i!= None]
        er_sqd.insert(0,None)
    print("For train set")
    info={"Y(t)":X_train,"Forecast":forecast1,"Errors":error,"Squared Errors":er_sqd}
    print(tabulate.tabulate(info,headers="keys",tablefmt="fancy_grid"))
    print("The mean squared error(MSE):",np.mean(er_sqd[1:]))
    
    print("The variance for prediction errors:",np.var(error[1:]))
     
    for i in range(len(X_test)):
        y = (X_train[-1]*alpha)+((1-alpha)*forecast1[-1])
        forecast2.append(y)
        value1 = X_test[i]-forecast2[0]
        error1.append(value1)
        er_sqd1= [i ** 2 for i in error1]
    h_step = pd.DataFrame(forecast2,columns=['y_hat']).set_index(X_test.index)
    print("For test set")
    info={"Y(t)":X_test,"Forecast":forecast2,"Errors":error1,"Squared Errors":er_sqd1}
    print(tabulate.tabulate(info,headers="keys",tablefmt="fancy_grid"))
    print("The mean squared error(MSE):",np.mean(er_sqd1))
    
    print("The Variance for the forecast error:",np.var(error1))
    
    fig, ax= plt.subplots()
    ax = ax or plt.gca()    
    ax.plot(X_test,label="Test Set")
    ax.plot(X_train,label="Train Set")
    ax.plot(h_step.y_hat,label="H-Step Forecast")
    ax.legend()
    ax.set_ylabel("Magnitude")
    ax.set_xlabel(f"Index\n alpha:{alpha}")
    ax.set_title("SES Method Forecasting")
    plt.xticks(rotation = 90)
    plt.show()
    if q_val==True:
        rk= stem_plot(error[2:],5)
        qv = len(X_train+X_test)*np.sum(np.square(rk[1:]))
        print("The Q-value for the training set:",qv)
    else:
        qv = " "
    pred_error_s = error[1:]
    for_error_s = error1
    if table == True:    
        er = np.mean(pred_error_s)
        vr= np.var(pred_error_s)
        vfr = np.var(for_error_s)
        af = {"Name of test": "SES", 'Q value':qv,'MSE for prediction errors': np.mean(er_sqd[1:len(X_train)]),"MSE for forecast errors":np.mean(er_sqd1),"Mean of prediction errors":er,"Variance of prediction errors": vr,"Variance of forecast errors": vfr,"Correlation Coeff":correlation_coefficent_cal(for_error_s,X_test)}
        tab = tab.append(af,ignore_index = True)
    display(tab)
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
# Moving average 3ma,4ma etc
def ma(order, folding=None):
    df[f"{order}-MA"] = " "
    if order == 1 or order == 2:
        print("m=1,2 will not be accepted")
    elif order % 2 == 0 and order != 2:
        folding = int(input("Enter the folding value"))
        df[f"{folding}X{order}-MA"] = " "
        l = math.floor(int((order - 1) / 2))
        for i in range(l, len(df)):
            if len(df.passengers[i - l:i - l + order]) == order:
                df[f"{order}-MA"][i] = (sum(df[i - l:i - l + order])) / order
        l = l + folding - 1
        for i in range(l, len(df) - l):
            df[f"{folding}X{order}-MA"][i] = (sum(
                df[f"{order}-MA"][df[f"{order}-MA"] != " "][i - l:(i - l + folding)])) / (folding)

    else:
        l = int((order - 1) / 2)
        for i in range(l, len(df) - l):
            df[f"{order}-MA"][i] = (sum(df[i - l:i - l + order])) / order
# %%
# gpac_for data
def arma_dat(df,na,nb):
    # N= int(input("Enter the Number of Samples:"))
    # mean_e  = int(input("Enter the mean for the WN"))
    # var_e  = int(input("Enter the Variance for the WN"))
    # e = np.random.normal(mean_e,var_e,N)
    # y  = np.zeros(len(e))
    # oa = int(input("Enter AR order"))
    # ob = int(input("Enter MA order"))
    # arparams = [1]
    # maparams = [1]
    # if oa>=1:
    #     for i in range(oa):
    #         ar = float(input(f"Enter the parameter a{i+1}"))
    #         arparams.append(ar)
    # if ob>=1:
    #     for i in range(ob):
    #         ma = float(input(f"Enter the parameter b{i+1}"))
    #         maparams.append(ma)
    # ar = np.r_[arparams]
    # ma = np.r_[maparams]
    # arma_process = sm.tsa.ArmaProcess(ar,ma)
    # print("Is this a stationary process:",arma_process.isstationary)
    # y = arma_process.generate_sample(N)
    lags = int(input("Enter the number of lags:"))
    # ry = arma_process.acf(lags=lags)
    # print(ry.shape)
    ry = stem_plot(df,lag =lags)
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
    ACF_PACF_Plot(df,lags = 20)
#%%
# ARMA_coeff_simulation.
def ARMA_coeff_simul():
    na = int(input("Enter the AR order:"))
    nb = int(input("Enter the MA order:"))
    arparams = [1]
    maparams = [1]
    lags = 20
    N = int(input("Enter the length of samples:"))
    if na >= 1:
        for i in range(na):
            ar = float(input(f"Enter the parameter a{i + 1}"))
            arparams.append(ar)
    if nb >= 1:
        for i in range(nb):
            ma = float(input(f"Enter the parameter b{i + 1}"))
            maparams.append(ma)
    ar = np.r_[arparams]
    ma = np.r_[maparams]
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    print("Is this a stationary process:", arma_process.isstationary)
    y = arma_process.generate_sample(N)

    # Calculate ACF
    ry = arma_process.acf(lags=lags)

    # Coefficient calculation
    model = sm.tsa.ARIMA(y, order=(na, 0, nb), trend="n", ).fit()
    for i in range(na):
        print("The AR coefficeint a{}".format(i + 1), "is", model.params[i] * -1)
    for i in range(nb):
        print("The MA coefficeint b{}".format(i + 1), "is", model.params[i + na])
    print(model.summary())

    # Prediction
    model_hat = model.predict(start=0, end=N - 1)
    e = y - model_hat
    re = stem_plot(e, lags, name="Errors")
    Q = len(y) * np.sum(np.square(re[lags:]))
    DOF = lags - na - nb
    alfa = 0.01
    chi_critical = chi2.ppf(1 - alfa, DOF)
    if Q < chi_critical:
        print("The residual is white")
    else:
        print("The residual is NOT white")
    lbvalue, pvalue = sm.stats.acorr_ljungbox(e, lags=[lags], return_df=True)
    print(lbvalue)
    print(pvalue)

    plt.figure()
    plt.plot(y, "r", label="True Data")

    plt.plot(model_hat, "b", label="Fitted Data")
    plt.xlabel("Samples")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title("statsmodel ARMA process generation, parameter estimation and prediction")
    plt.show()
# %%
# ARMA coeff simul for general data
def ARMA_coeff_simul_dat(df):
    na = int(input("Enter the AR order:"))
    nb = int(input("Enter the MA order:"))
    # arparams = [1]
    # maparams = [1]
    # lags = 20
    # N = int(input("Enter the length of samples:"))
    # if na >= 1:
    #     for i in range(na):
    #         ar = float(input(f"Enter the parameter a{i + 1}"))
    #         arparams.append(ar)
    # if nb >= 1:
    #     for i in range(nb):
    #         ma = float(input(f"Enter the parameter b{i + 1}"))
    #         maparams.append(ma)
    # ar = np.r_[arparams]
    # ma = np.r_[maparams]
    # arma_process = sm.tsa.ArmaProcess(ar, ma)
    # print("Is this a stationary process:", arma_process.isstationary)
    y = df
    # Calculate ACF
    lags = int(input("Enter the number of lags"))
    ry = stem_plot(y,lags)

    # Coefficient calculation
    model = sm.tsa.ARIMA(y, order=(na, 0, nb), trend="n", ).fit()
    for i in range(na):
        print("The AR coefficeint a{}".format(i + 1), "is", model.params[i] * -1)
    for i in range(nb):
        print("The MA coefficeint b{}".format(i + 1), "is", model.params[i + na])
    print(model.summary())

    # Prediction
    model_hat = model.predict(start=0, end=len(train) - 1)
    e = y[1:] - model_hat[:-1]
    re = stem_plot(e, lags, name="Errors")
    Q = len(y) * np.sum(np.square(re[lags:]))
    DOF = lags - na - nb
    alfa = 0.01
    chi_critical = chi2.ppf(1 - alfa, DOF)
    if Q < chi_critical:
        print("The residual is white")
    else:
        print("The residual is NOT white")
    lbvalue, pvalue = sm.stats.acorr_ljungbox(e, lags=[lags], return_df=True)
    print(lbvalue)
    print(pvalue)

    plt.figure()
    plt.plot(y, "r", label="True Data")

    plt.plot(model_hat, "b", label="Fitted Data")
    plt.xlabel("Samples")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title("statsmodel ARMA process generation, parameter estimation and prediction")
    plt.show()
    return model_hat,e
# %%
def LSE():
    np.random.seed(42)
    na = int(input("Enter the AR order:"))
    nb = int(input("Enter the MA order:"))
    arparams = [1]
    maparams = [1]
    if na >= 1:
        for i in range(na):
            ar = float(input(f"Enter the parameter a{i + 1}"))
            arparams.append(ar)
    if nb >= 1:
        for i in range(nb):
            ma = float(input(f"Enter the parameter b{i + 1}"))
            maparams.append(ma)
    tr = int(input("Enter the train set %:"))
    ts = int(input("Enter the test set %:"))
    ar = np.r_[arparams]
    ma = np.r_[maparams]
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    print("Is this a stationary process:", arma_process.isstationary)
    y = arma_process.generate_sample(10000)
    # Step 0
    theta = np.zeros(shape=(1, na + nb))
    num_d = [1]
    den_d = [1]
    num = np.r_[1, [0] * (na + nb - 2)]
    den = np.r_[1, [0] * (na + nb - 2)]
    # Step 1
    mu = 0.01
    N = len(y)
    xi = np.zeros((N, na + nb))
    delta = (10 ** -6)
    sse_g = []
    it = []
    sse = []
    A = []
    g = []
    I = []
    sse = []
    sse_new = []
    theta_new = []
    c_theta = []
    delta = 10 ** -6
    # Step1
    if nb == na:
        den = np.concatenate((den_d, theta[0, na:]))
        num = np.concatenate((num_d, theta[0, :na]))
    elif na > nb:
        den = np.concatenate((den_d, theta[0, na:]))
        den = np.concatenate((den, np.zeros(na - nb)))
        num = np.concatenate((num_d, theta[0, :na]))
    elif nb > na:
        num = np.concatenate((num_d, theta[0, :na]))
        num = np.concatenate((num, np.zeros(nb - na)))
        den = np.concatenate((den_d, theta[0, na:]))
    sys = (num, den, 1)
    _, e = signal.dlsim(sys, y)
    # print(theta,"\n",num,"\n",den)
    for i in range(na + nb):
        theta[0, i] = theta[0, i] + delta

        if nb == na:
            den = np.concatenate((den_d, theta[0, na:]))
            num = np.concatenate((num_d, theta[0, :na]))
        elif na > nb:
            den = np.concatenate((den_d, theta[0, na:]))
            den = np.concatenate((den, np.zeros(na - nb)))
            num = np.concatenate((num_d, theta[0, :na]))
        elif nb > na:
            num = np.concatenate((num_d, theta[0, :na]))
            num = np.concatenate((num, np.zeros(nb - na)))
            den = np.concatenate((den_d, theta[0, na:]))
        sys = (num, den, 1)
        _, e1 = signal.dlsim(sys, y)
        theta[0, i] = theta[0, i] - delta
        xi[:, i] = ((e - e1) / delta)[:, 0]
    sse = np.matmul(e1.T, e1)
    A = np.matmul(xi.T, xi)
    g = np.matmul(xi.T, e)
    # Step 2
    I = np.identity(na + nb)
    c_theta = np.matmul(np.linalg.pinv(A + (mu * I)), g)
    theta_new = theta + c_theta.T
    if nb == na:
        den = np.concatenate((den_d, theta_new[0, na:]))
        num = np.concatenate((num_d, theta_new[0, :na]))
    elif na > nb:
        den = np.concatenate((den_d, theta_new[0, na:]))
        den = np.concatenate((den, np.zeros(na - nb)))
        num = np.concatenate((num_d, theta_new[0, :na]))
    elif nb > na:
        num = np.concatenate((num_d, theta_new[0, :na]))
        num = np.concatenate((num, np.zeros(nb - na)))
        den = np.concatenate((den_d, theta_new[0, na:]))
    sys = (num, den, 1)
    _, e2 = signal.dlsim(sys, y)
    sse_new = np.matmul(e2.T, e2)
    # Step 3
    max_iter = 50
    for i in range(max_iter):
        # print(f"======ITERATION {i}=======")
        if sse_new[0, 0] < sse[0, 0]:
            if np.linalg.norm(c_theta) < 10 ** -3:
                theta = theta_new
                var = sse_new[0, 0] / (N - (na + nb))
                cov = var * np.linalg.pinv(A)
            else:
                theta = theta_new
                mu = mu / 10
                # Step1
                if nb == na:
                    den = np.concatenate((den_d, theta[0, na:]))
                    num = np.concatenate((num_d, theta[0, :na]))
                elif na > nb:
                    den = np.concatenate((den_d, theta[0, na:]))
                    den = np.concatenate((den, np.zeros(na - nb)))
                    num = np.concatenate((num_d, theta[0, :na]))
                elif nb > na:
                    num = np.concatenate((num_d, theta[0, :na]))
                    num = np.concatenate((num, np.zeros(nb - na)))
                    den = np.concatenate((den_d, theta[0, na:]))
                sys = (num, den, 1)
                _, e = signal.dlsim(sys, y)
                # print(theta,"\n",num,"\n",den)
                for i in range(na + nb):
                    theta[0, i] = theta[0, i] + delta

                    if nb == na:
                        den = np.concatenate((den_d, theta[0, na:]))
                        num = np.concatenate((num_d, theta[0, :na]))
                    elif na > nb:
                        den = np.concatenate((den_d, theta[0, na:]))
                        den = np.concatenate((den, np.zeros(na - nb)))
                        num = np.concatenate((num_d, theta[0, :na]))
                    elif nb > na:
                        num = np.concatenate((num_d, theta[0, :na]))
                        num = np.concatenate((num, np.zeros(nb - na)))
                        den = np.concatenate((den_d, theta[0, na:]))
                    sys = (num, den, 1)
                    _, e1 = signal.dlsim(sys, y)
                    theta[0, i] = theta[0, i] - delta
                    xi[:, i] = ((e - e1) / delta)[:, 0]
                sse = np.matmul(e1.T, e1)
                A = np.matmul(xi.T, xi)
                g = np.matmul(xi.T, e)
                # step 2
                I = np.identity(na + nb)
                c_theta = np.matmul(np.linalg.pinv(A + (mu * I)), g)
                theta_new = theta + c_theta.T
                if nb == na:
                    den = np.concatenate((den_d, theta_new[0, na:]))
                    num = np.concatenate((num_d, theta_new[0, :na]))
                elif na > nb:
                    den = np.concatenate((den_d, theta_new[0, na:]))
                    den = np.concatenate((den, np.zeros(na - nb)))
                    num = np.concatenate((num_d, theta_new[0, :na]))
                elif nb > na:
                    num = np.concatenate((num_d, theta_new[0, :na]))
                    num = np.concatenate((num, np.zeros(nb - na)))
                    den = np.concatenate((den_d, theta_new[0, na:]))
                sys = (num, den, 1)
                _, e2 = signal.dlsim(sys, y)
                sse_new = np.matmul(e2.T, e2)
        while sse_new > sse:
            mu = mu * 10
            if mu > 10 ** 12:
                print("Error:", theta)
            # step 2
            I = np.identity(na + nb)
            c_theta = np.matmul(np.linalg.pinv(A + (mu * I)), g)
            theta_new = theta + c_theta.T
            if nb == na:
                den = np.concatenate((den_d, theta_new[0, na:]))
                num = np.concatenate((num_d, theta_new[0, :na]))
            elif na > nb:
                den = np.concatenate((den_d, theta_new[0, na:]))
                den = np.concatenate((den, np.zeros(na - nb)))
                num = np.concatenate((num_d, theta_new[0, :na]))
            elif nb > na:
                num = np.concatenate((num_d, theta_new[0, :na]))
                num = np.concatenate((num, np.zeros(nb - na)))
                den = np.concatenate((den_d, theta_new[0, na:]))
            sys = (num, den, 1)
            _, e2 = signal.dlsim(sys, y)
            sse_new = np.matmul(e2.T, e2)
        sse_g.append(sse[0, 0])
        it.append(i + 1)
        # theta = theta_new
    print(theta)
    plt.plot(sse_g)
    plt.title("SSE vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("SSE")
    plt.show()
    print("The confidence interval:\n")
    for i in range(na):
        print(f"{theta[0, i] - np.sqrt(cov[i, i])}<a{i + 1}<{theta[0, i] + np.sqrt(cov[i, i])}\n")
    for i in range(nb):
        print(f"{theta[0, i + na] - np.sqrt(cov[i, i])}<b{i + 1}<{theta[0, i + na] + np.sqrt(cov[i, i])}\n")
    # e = np.random.normal(0,1,10000)
    # tf = (num,den,1)
    # y = signal.dlsim(tf,e)
    y_train, y_test = train_test_split(y, train_size=tr / 100, test_size=ts / 100)
    return theta, y, e, y_train, y_test, cov, num, den, tr, ts
#%%
# LM algorithm for general data
def LSE_dats(df):
    np.random.seed(42)
    na = int(input("Enter the AR order:"))
    nb = int(input("Enter the MA order:"))
    # arparams = [1]
    # maparams = [1]
    # if na >= 1:
    #     for i in range(na):
    #         ar = float(input(f"Enter the parameter a{i + 1}"))
    #         arparams.append(ar)
    # if nb >= 1:
    #     for i in range(nb):
    #         ma = float(input(f"Enter the parameter b{i + 1}"))
    #         maparams.append(ma)
    tr = int(input("Enter the train set %:"))
    ts = int(input("Enter the test set %:"))
    # ar = np.r_[arparams]
    # ma = np.r_[maparams]
    # arma_process = sm.tsa.ArmaProcess(ar, ma)
    # print("Is this a stationary process:", arma_process.isstationary)
    # y = arma_process.generate_sample(10000)
    y = df
    # Step 0
    theta = np.zeros(shape=(1, na + nb))
    num_d = [1]
    den_d = [1]
    num = np.r_[1, [0] * (na + nb - 2)]
    den = np.r_[1, [0] * (na + nb - 2)]
    # Step 1
    mu = 0.01
    N = len(y)
    xi = np.zeros((N, na + nb))
    delta = (10 ** -6)
    sse_g = []
    it = []
    sse = []
    A = []
    g = []
    I = []
    sse = []
    sse_new = []
    theta_new = []
    c_theta = []
    delta = 10 ** -6
    # Step1
    if nb == na:
        den = np.concatenate((den_d, theta[0, na:]))
        num = np.concatenate((num_d, theta[0, :na]))
    elif na > nb:
        den = np.concatenate((den_d, theta[0, na:]))
        den = np.concatenate((den, np.zeros(na - nb)))
        num = np.concatenate((num_d, theta[0, :na]))
    elif nb > na:
        num = np.concatenate((num_d, theta[0, :na]))
        num = np.concatenate((num, np.zeros(nb - na)))
        den = np.concatenate((den_d, theta[0, na:]))
    sys = (num, den, 1)
    _, e = signal.dlsim(sys, y)
    # print(theta,"\n",num,"\n",den)
    for i in range(na + nb):
        theta[0, i] = theta[0, i] + delta

        if nb == na:
            den = np.concatenate((den_d, theta[0, na:]))
            num = np.concatenate((num_d, theta[0, :na]))
        elif na > nb:
            den = np.concatenate((den_d, theta[0, na:]))
            den = np.concatenate((den, np.zeros(na - nb)))
            num = np.concatenate((num_d, theta[0, :na]))
        elif nb > na:
            num = np.concatenate((num_d, theta[0, :na]))
            num = np.concatenate((num, np.zeros(nb - na)))
            den = np.concatenate((den_d, theta[0, na:]))
        sys = (num, den, 1)
        _, e1 = signal.dlsim(sys, y)
        theta[0, i] = theta[0, i] - delta
        xi[:, i] = ((e - e1) / delta)[:, 0]
    sse = np.matmul(e1.T, e1)
    A = np.matmul(xi.T, xi)
    g = np.matmul(xi.T, e)
    # Step 2
    I = np.identity(na + nb)
    c_theta = np.matmul(np.linalg.pinv(A + (mu * I)), g)
    theta_new = theta + c_theta.T
    if nb == na:
        den = np.concatenate((den_d, theta_new[0, na:]))
        num = np.concatenate((num_d, theta_new[0, :na]))
    elif na > nb:
        den = np.concatenate((den_d, theta_new[0, na:]))
        den = np.concatenate((den, np.zeros(na - nb)))
        num = np.concatenate((num_d, theta_new[0, :na]))
    elif nb > na:
        num = np.concatenate((num_d, theta_new[0, :na]))
        num = np.concatenate((num, np.zeros(nb - na)))
        den = np.concatenate((den_d, theta_new[0, na:]))
    sys = (num, den, 1)
    _, e2 = signal.dlsim(sys, y)
    sse_new = np.matmul(e2.T, e2)
    # Step 3
    max_iter = 50
    for i in range(max_iter):
        # print(f"======ITERATION {i}=======")
        if sse_new[0, 0] < sse[0, 0]:
            if np.linalg.norm(c_theta) < 10 ** -3:
                theta = theta_new
                var = sse_new[0, 0] / (N - (na + nb))
                cov = var * np.linalg.pinv(A)
            else:
                theta = theta_new
                mu = mu / 10
                # Step1
                if nb == na:
                    den = np.concatenate((den_d, theta[0, na:]))
                    num = np.concatenate((num_d, theta[0, :na]))
                elif na > nb:
                    den = np.concatenate((den_d, theta[0, na:]))
                    den = np.concatenate((den, np.zeros(na - nb)))
                    num = np.concatenate((num_d, theta[0, :na]))
                elif nb > na:
                    num = np.concatenate((num_d, theta[0, :na]))
                    num = np.concatenate((num, np.zeros(nb - na)))
                    den = np.concatenate((den_d, theta[0, na:]))
                sys = (num, den, 1)
                _, e = signal.dlsim(sys, y)
                # print(theta,"\n",num,"\n",den)
                for i in range(na + nb):
                    theta[0, i] = theta[0, i] + delta

                    if nb == na:
                        den = np.concatenate((den_d, theta[0, na:]))
                        num = np.concatenate((num_d, theta[0, :na]))
                    elif na > nb:
                        den = np.concatenate((den_d, theta[0, na:]))
                        den = np.concatenate((den, np.zeros(na - nb)))
                        num = np.concatenate((num_d, theta[0, :na]))
                    elif nb > na:
                        num = np.concatenate((num_d, theta[0, :na]))
                        num = np.concatenate((num, np.zeros(nb - na)))
                        den = np.concatenate((den_d, theta[0, na:]))
                    sys = (num, den, 1)
                    _, e1 = signal.dlsim(sys, y)
                    theta[0, i] = theta[0, i] - delta
                    xi[:, i] = ((e - e1) / delta)[:, 0]
                sse = np.matmul(e1.T, e1)
                A = np.matmul(xi.T, xi)
                g = np.matmul(xi.T, e)
                # step 2
                I = np.identity(na + nb)
                c_theta = np.matmul(np.linalg.pinv(A + (mu * I)), g)
                theta_new = theta + c_theta.T
                if nb == na:
                    den = np.concatenate((den_d, theta_new[0, na:]))
                    num = np.concatenate((num_d, theta_new[0, :na]))
                elif na > nb:
                    den = np.concatenate((den_d, theta_new[0, na:]))
                    den = np.concatenate((den, np.zeros(na - nb)))
                    num = np.concatenate((num_d, theta_new[0, :na]))
                elif nb > na:
                    num = np.concatenate((num_d, theta_new[0, :na]))
                    num = np.concatenate((num, np.zeros(nb - na)))
                    den = np.concatenate((den_d, theta_new[0, na:]))
                sys = (num, den, 1)
                _, e2 = signal.dlsim(sys, y)
                sse_new = np.matmul(e2.T, e2)
        while sse_new > sse:
            mu = mu * 10
            if mu > 10 ** 12:
                print("Error:", theta)
            # step 2
            I = np.identity(na + nb)
            c_theta = np.matmul(np.linalg.pinv(A + (mu * I)), g)
            theta_new = theta + c_theta.T
            if nb == na:
                den = np.concatenate((den_d, theta_new[0, na:]))
                num = np.concatenate((num_d, theta_new[0, :na]))
            elif na > nb:
                den = np.concatenate((den_d, theta_new[0, na:]))
                den = np.concatenate((den, np.zeros(na - nb)))
                num = np.concatenate((num_d, theta_new[0, :na]))
            elif nb > na:
                num = np.concatenate((num_d, theta_new[0, :na]))
                num = np.concatenate((num, np.zeros(nb - na)))
                den = np.concatenate((den_d, theta_new[0, na:]))
            sys = (num, den, 1)
            _, e2 = signal.dlsim(sys, y)
            sse_new = np.matmul(e2.T, e2)
        sse_g.append(sse[0, 0])
        it.append(i + 1)
        # theta = theta_new
    print(theta)
    plt.plot(sse_g)
    plt.title("SSE vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("SSE")
    plt.show()
    print("The confidence interval:\n")
    for i in range(na):
        print(f"{theta[0, i] - np.sqrt(cov[i, i])}<a{i + 1}<{theta[0, i] + np.sqrt(cov[i, i])}\n")
    for i in range(nb):
        print(f"{theta[0, i + na] - np.sqrt(cov[i, i])}<b{i + 1}<{theta[0, i + na] + np.sqrt(cov[i, i])}\n")
    # e = np.random.normal(0,1,10000)
    # tf = (num,den,1)
    # y = signal.dlsim(tf,e)
    y_train, y_test = train_test_split(y, train_size=tr / 100, test_size=ts / 100)
    return theta, y, e, y_train, y_test, cov, num, den, tr, ts
#%%
# ACF & PACF
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
# log transformation
def log_trans(column):
    lg= []
    for i in range(len(column)):
        value = math.log(column[i])
        lg.append(value)
    return lg
# %%
# differencing
def difference(dataset, interval =1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i]- dataset[i-interval]
        diff.append(value)
    return diff
# %%
# # SES alternative
# from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
# y_hat_avg = test.copy()
# fit2 = SimpleExpSmoothing(np.asarray(train['Count'])).fit(smoothing_level=0.6,optimized=False)
# y_hat_avg['SES'] = fit2.forecast(len(test))
# plt.figure(figsize=(16,8))
# plt.plot(train['Count'], label='Train')
# plt.plot(test['Count'], label='Test')
# plt.plot(y_hat_avg['SES'], label='SES')
# plt.legend(loc='best')
# plt.show()
#%%
# # Holt winter
# y_hat_avg = test.copy()
# fit1 = ExponentialSmoothing(np.asarray(train['Count']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
# y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
# plt.figure(figsize=(16,8))
# plt.plot( train['Count'], label='Train')
# plt.plot(test['Count'], label='Test')
# plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
# plt.legend(loc='best')
# plt.show()
#%%
# # Sarima
# # y_hat_avg = test.copy()
# fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
# y_hat_avg['SARIMA'] = fit1.predict(start="2013-11-1", end="2013-12-31", dynamic=True)
# plt.figure(figsize=(16,8))
# plt.plot( train['Count'], label='Train')
# plt.plot(test['Count'], label='Test')
# plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
# plt.legend(loc='best')
# plt.show()
#%%
# holt Linear trend
# y_hat_avg = test.copy()

# fit1 = Holt(np.asarray(train['Count'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
# y_hat_avg['Holt_linear'] = fit1.forecast(len(test))

# plt.figure(figsize=(16,8))
# plt.plot(train['Count'], label='Train')
# plt.plot(test['Count'], label='Test')
# plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
# plt.legend(loc='best')
# plt.show()
# #%%
# def cat_var_checker(df, dtype='object'):
#     # Get the dataframe of categorical variables and their number of unique value
#     df_cat = pd.DataFrame([[var, df[var].nunique(dropna=False)]
#                            # If the data type is dtype
#                            for var in df.columns if df[var].dtype == dtype],
#                           columns=['var', 'nunique'])
    
#     # Sort df_cat in accending order of the number of unique value
#     df_cat = df_cat.sort_values(by='nunique', ascending=False).reset_index(drop=True)
    
#     return df_cat
# # %%
# def common_var_checker(df_train, df_val, df_test, target):
#     # Get the dataframe of common variables between the training, validation and test data
#     df_common_var = pd.DataFrame(np.intersect1d(np.intersect1d(df_train.columns, df_val.columns), np.union1d(df_test.columns, [target])),
#                                  columns=['common var'])
                
#     return df_common_var
# %%
