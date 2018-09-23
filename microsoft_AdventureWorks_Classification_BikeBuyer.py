import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import xlrd
import math
import seaborn as sns
import csv



#Paths
AdvWorksdat= '\\Adventure Works Bicycle Data.xlsx'

dataset=  pd.read_excel(AdvWorksdat, sheet_name='Cust_Bike_buy_coombine_edit_dat', index_col='CustomerID',na_values='?' )
dataset2= pd.read_excel(AdvWorksdat, sheet_name='AW_test (2)', index_col='CustomerID',na_values='?' )



unwantedcols_train= ['State','Country']    #'BirthDate',
unwantedcols_test= ['BirthDate', 'StateProvinceName','CountryRegionName']


dataset.drop(axis=1, columns=unwantedcols_train, inplace=True)
dataset2.drop(axis=1, columns=unwantedcols_test, inplace=True)



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
    for col in objcols:
        dataset[col] = dataset[col].cat.codes


    #fill NA
    for col in numcols:
        dataset.fillna(dataset[col].mean, inplace=True)


    #reading y and appending to last
    if('BikeBuyer' in dataset.columns):
        temp=dataset['BikeBuyer']
        dataset.drop(axis=1, columns='BikeBuyer', inplace=True)
        new_dataset= dataset.join(temp, on='CustomerID')
    else:
        new_dataset=dataset

    return new_dataset


traindataset= clean_process(dataset)
print('================================')
testdataset= clean_process(dataset2)
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


from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)
y_pred_ver= rf.predict(X_test)

y_pred= rf.predict(X_test_act)

with open('\\Bikebuyer_resuts.csv', 'w') as res:
    wr = csv.writer(res)
    wr.writerow(rf.predict(X_test_act))

#Classification Metrics
def class_metrics(labels, scores):
    import sklearn.metrics as sklm
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('====================================================')
    print('                 Score positive    Score negative')
    print('----------------------------------------------------')
    print('Actual positive    %6d' % conf[0, 0] + '             %5d' % conf[0, 1])
    print('Actual negative    %6d' % conf[1, 0] + '             %5d' % conf[1, 1])
    print('')
    print('----------------------------------------------------')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))    #Acc = (TP+TN)/(TP+FP+TN+FN)
    print(' ')
    print('==================================================')
    print('           Positive      Negative')
    print('----------------------------------------------------')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])   #F1 is a weighted metric for overall model performance

class_metrics(y_test, rf.predict(X_test))


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classification Visualization of Predicted set')
plt.xlabel('Features')
plt.ylabel('Bike Buyer')
plt.legend()
plt.show()