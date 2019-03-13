#### Kerela Rainfall Predict ######
### Dataset - Sub_Division_IMD_2017  ####


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import math
import scipy.stats as ss


org_dataset_path= '\\Sub_Division_IMD_2017.csv'


org_dataset= pd.read_csv(org_dataset_path)


#print(org_dataset.head())
# print(org_dataset.shape)
#print(set(org_dataset.SUBDIVISION))
print('================================================================================')


kerala_dataset= org_dataset[org_dataset.SUBDIVISION=='Kerala']

# print(kerala_dataset.head())
# print(kerala_dataset.shape)
# print('================================================================================')

kerala_dataset.drop(labels='SUBDIVISION',axis=1,inplace=True)
kerala_dataset.set_index('YEAR',inplace=True)
#kerala_dataset.set_index(keys='SUBDIVISION',inplace=True)
#print(kerala_dataset)
# print(kerala_dataset.shape)
# print('-------------------------------------------------')
# print(kerala_dataset.info())
# print('-------------------------------------------------')
# print(kerala_dataset.describe())
# print('-------------------------------------------------')


#
# ##Mising data percent by col code
# for col in kerala_dataset.columns:
#     print('For ', col, 'Missing value percent is = ', kerala_dataset[col].isnull().count()/len(kerala_dataset), '\t NA value percent is = ', kerala_dataset[col].isna().count() / len(kerala_dataset) )
for col in kerala_dataset.columns:
    kerala_dataset.fillna(kerala_dataset[col].mean(), inplace=True)

##Outliner remove
for col in kerala_dataset.columns:
    limit = kerala_dataset[col].quantile(0.95)
    kerala_dataset[col] = kerala_dataset[col].mask(kerala_dataset[col] > limit, limit)



kerala_monthly = kerala_dataset.iloc[:, 1:11]

## Check Stationary of data
def stationarygraph(data, roll_win=10):
    rollmean= data.groupby('YEAR').mean()['ANNUAL'].rolling(roll_win).mean()
    rollstd= data.groupby('YEAR').mean()['ANNUAL'].rolling(roll_win).std()

    orig=plt.plot(data['ANNUAL'], color='blue', label='Original')
    mean=plt.plot(rollmean, color='red', label='Rolling Mean')
    std=plt.plot(rollstd, color='red', label='Rolling STD')
    plt.legend(loc='best')
    plt.title('Kerala Annual Rainfall from Year 1901 to 2017')
    plt.show()

def dickerytest(dataset, title):
    ##perform Dickery-Fuller Test
    from statsmodels.tsa.stattools import adfuller
    print('----------------------------------------')
    print('Result of Dickery test for ', title, ' :')
    dftest= adfuller(dataset['ANNUAL'], autolag='AIC')
    dfout= pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key,value in dftest[4].items():
        dfout['Critical Value (%s)'%key]=value

    print(dfout)
    print('----------------------------------------')

stationarygraph(kerala_dataset)
dickerytest(kerala_dataset, 'Initial')

##Moving Average
log_kerala= np.log(kerala_dataset)
log_mov_avg=log_kerala.groupby('YEAR').mean()['ANNUAL'].rolling(10).mean()
plt.plot(log_kerala['ANNUAL'])
plt.plot(log_mov_avg, color='red')
plt.title('LOG transformed Visualization')
plt.show()

log_mov_avg_diff= log_kerala['ANNUAL']-log_mov_avg
log_mov_avg_diff_dataset=pd.DataFrame({'YEAR':log_mov_avg_diff.index ,'ANNUAL':log_mov_avg_diff.values})

log_mov_avg_diff_dataset['ANNUAL'].dropna(inplace=True)
dickerytest(log_mov_avg_diff_dataset, 'Moving Average')


##Eliminating Trend and Seasonality
##differencing
diff_ker_data= log_mov_avg_diff_dataset['ANNUAL']-log_mov_avg_diff_dataset['ANNUAL'].shift()
diff_ker_dataset= pd.DataFrame({'YEAR':diff_ker_data.index ,'ANNUAL':diff_ker_data.values})

plt.plot(diff_ker_dataset)
plt.title('Differencing Visualization')
plt.show()

diff_ker_dataset['ANNUAL'].dropna(inplace=True)
dickerytest(diff_ker_dataset, 'Differncing')




##ACF and PACF plots
from statsmodels.tsa.stattools import acf,pacf

lag_acf= acf(diff_ker_dataset['ANNUAL'], nlags=10)
lag_pacf= pacf(diff_ker_dataset['ANNUAL'], nlags=10, method='ols')

##plot ACF  --  q value - obtained 1
plt.plot(lag_acf)
plt.axhline(y=0, linestyle= '--', color='blue')
plt.axhline(y=-1.96/np.sqrt(len(diff_ker_dataset)),linestyle= '--',color= 'red')
plt.axhline(y=1.96/np.sqrt(len(diff_ker_dataset)),linestyle= '--',color= 'green')
plt.title('ACF')
plt.show()

##plot PACF  --  p value - obtained 1
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle= '--', color='blue')
plt.axhline(y=-1.96/np.sqrt(len(diff_ker_dataset)),linestyle= '--',color= 'red')
plt.axhline(y=1.96/np.sqrt(len(diff_ker_dataset)),linestyle= '--',color= 'green')
plt.title('PACF')
plt.show()


###ARIMA model

from statsmodels.tsa.arima_model import ARIMA
model1= ARIMA(log_mov_avg_diff_dataset['ANNUAL'], order=(1,1,1))
result1 = model1.fit(disp=-1)
plt.plot(diff_ker_dataset['ANNUAL'])
plt.plot(result1.fittedvalues, color='red')
plt.title('ARIMA')
plt.show()

##predicted results without any back transformation
pred_arima_diff= pd.Series(result1.fittedvalues, copy=True)
print('----------------------------------------')
print('Original Predicted values: \n', pred_arima_diff.head())
print('----------------------------------------')

##cumulative sum
pred_arima_diff_cum= pred_arima_diff.cumsum()
print('----------------------------------------')
print('Cumulative Sum of  Predicted values: \n', pred_arima_diff_cum.head())
print('----------------------------------------')

##add difference
pred_arima_diff_log= pd.Series(log_mov_avg_diff_dataset.ix[0], index=log_mov_avg_diff_dataset['YEAR'])
pred_arima_diff_log= pred_arima_diff_log.add(pred_arima_diff_cum, fill_value=0)
print('----------------------------------------')
print('Adding back difference to Predicted values: \n', pred_arima_diff_log.head())
print('----------------------------------------')


##exponential
pred_arima= np.exp(pred_arima_diff_log)
print('----------------------------------------')
print('Exponential of Predicted values: \n',pred_arima)
print('----------------------------------------')
plt.plot(log_mov_avg_diff_dataset['ANNUAL'], color= 'blue')
plt.plot(pred_arima, color='red')
plt.title('Predict vs Actual')
plt.show()

