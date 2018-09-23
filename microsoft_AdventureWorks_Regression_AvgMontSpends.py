import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import xlrd
import math
import seaborn as sns
import csv



#Paths
AdvWorksdat= '\\Adventure Works Bicycle Data\\Adventure Works Bicycle Data.xlsx'

dataset=  pd.read_excel(AdvWorksdat, sheet_name='Cust_AvgMonthspen_comb_data (2)', index_col='CustomerID',na_values='?' )
dataset2= pd.read_excel(AdvWorksdat, sheet_name='AW_test (2)', index_col='CustomerID',na_values='?' )



unwantedcols= ['BirthDate', 'StateProvinceName','CountryRegionName']


dataset.drop(axis=1, columns=unwantedcols, inplace=True)
dataset2.drop(axis=1, columns=unwantedcols, inplace=True)



#Clean data function
def clean_process(dataset):
    objcols = []
    numcols = []
    #getting numcols and objcols
    for col in dataset.columns:
        if dataset[col].dtypes == 'object':
            objcols.append(col)
        elif dataset[col].dtypes in ['int32', 'float32', 'int64', 'float64']:
            numcols.append(col)


    dataset[objcols] = dataset[objcols].astype('category')
    #dataset= pd.get_dummies(dataset, prefix='dum',prefix_sep='_',columns=objcols,drop_first=True,sparse=True,dtype=int)
    for col in objcols:
        dataset[col] = dataset[col].cat.codes



    #fill NA
    for col in numcols:
        dataset.fillna(dataset[col].mean, inplace=True)


    #reading y and appending to last
    if('AveMonthSpend' in dataset.columns):
        temp=dataset['AveMonthSpend']
        dataset.drop(axis=1, columns='AveMonthSpend', inplace=True)
        new_dataset= dataset.join(temp, on='CustomerID')
    else:
        new_dataset=dataset

    return new_dataset


traindataset= clean_process(dataset)
#print(traindataset.dtypes)
#print(traindataset.shape)
print('================================')
testdataset= clean_process(dataset2)
#print(testdataset.dtypes)
#print(testdataset.shape)
print('================================')


X= traindataset.iloc[:, :-1].values
y= traindataset.iloc[:, -1].values

X_test_act= testdataset.iloc[:,:].values



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit_transform(X)
sc.fit_transform(X_test_act)


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train, y_train)
y_pred_ver= lr.predict(X_test)

y_pred= lr.predict(X_test_act)


with open('\\Adventure Works Bicycle Data\\result.csv', 'w') as res:
    wr = csv.writer(res)
    wr.writerow(lr.predict(X_test_act))


#Evaluate LinearModel
def Linmodel_metrics(y_predicted, n_parameters,y_true=y_test):
    import sklearn.metrics as met
    ## First compute R^2 and the adjusted R^2
    r2 = met.r2_score(y_true, y_predicted)
    r2_adj = r2 - (n_parameters - 1) / (y_true.shape[0] - n_parameters) * (1 - r2)

    ## Print the usual metrics and the R^2 values
    print('Mean Square Error      = ' + str(met.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(met.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(met.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(met.median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))
    print('Adjusted R^2           = ' + str(r2_adj))

Linmodel_metrics(y_pred_ver, 11)




