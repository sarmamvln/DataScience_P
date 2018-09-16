import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy
import sklearn

trainpath= '\\WNS Analytics Wizard  2018_train_LZdllcl.csv'

#Actual Test data path
testpath= '\\WNS Analytics Wizard  2018_test_2umaH9m.csv'

colsname=['employee_id', 'department', 'region', 'education', 'gender', 'recruitment_channel',  'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'KPIs_met', 'awards_won', 'avg_training_score', 'is_promoted']

#Train Dataset
traindataset= pd.read_csv(trainpath, header=None, names=colsname, index_col='employee_id',na_values='!')  #

#Actual test data read
testdataset= pd.read_csv(testpath, header=None, names=colsname[0:13], index_col='employee_id',na_values='!') #


objcols= ['department', 'region', 'education', 'gender', 'recruitment_channel']
numcols= ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'KPIs_met', 'awards_won', 'avg_training_score']


# #Count unique values in dataset
# def count_unique(dataset, cols):
#     for col in cols:
#         print('----------------------------')
#         print('\n' + 'For column ' + col)
#         print(dataset[col].value_counts())
#
#
# print('Train dataset unique count: ')
# print('==============================')
# count_unique(traindataset, objcols)
#
# print('================================================================')
# #Actual Test data unique counts
# print('Actual Test dataset unique count: ')
# print('==============================')
# count_unique(testdataset, objcols)
# print('=================================================================')


#Clean data function
def clean_process(dataset, objcols= objcols, numcols=numcols):
    dataset.dropna(axis=0,how='any',inplace=True)
    for column in numcols:
        dataset[column] = pd.to_numeric(dataset[column],errors='coerce')
        if('is_promoted' in dataset.columns ):
            dataset['is_promoted'] = pd.to_numeric(dataset['is_promoted'], errors='coerce')



    dataset[objcols] = dataset[objcols].astype('category')
    dataset['dum_department'] = dataset['department'].cat.codes
    dataset['dum_region'] = dataset['region'].cat.codes
    dataset['dum_education'] = dataset['education'].cat.codes
    dataset['dum_gender'] = dataset['gender'].cat.codes
    dataset['dum_recruitment_channel'] = dataset['recruitment_channel'].cat.codes


    dataset= dataset.drop(axis=1, columns=objcols)

    # for col in numcols:
    #     dataset[col] = dataset[col].fillna(dataset[col].mean())


    return dataset



dataset= clean_process(traindataset)

#Actual Testdata cleaning
dataset_tes= clean_process(testdataset)


# #Bar chart
# def plot_bars(dataset, cols):
#     for col in cols:
#         fig = plt.figure(figsize=(6,6))
#         ax = fig.gca()
#         counts = dataset[col].value_counts()
#         counts.plot.bar(ax = ax, color = 'blue')
#         ax.set_title('Number of Prometed by ' + col)
#         ax.set_xlabel(col)
#         ax.set_ylabel('Count of Promoted')
#         plt.show()
#
# plot_bars(dataset, num_cols)



# print(dataset.dtypes)
# print('================================================')
#Actual Test data dtypes
#print(dataset_tes.dtypes)


X= dataset.loc[:, ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'KPIs_met', 'awards_won', 'avg_training_score','dum_department','dum_region','dum_education','dum_gender','dum_recruitment_channel']].values
y= dataset.loc[:, 'is_promoted'].values



#for validating model accuracy
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#for actual test data validating
#X_test= dataset_tes.iloc[:].values

from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

dc.fit(X_train, y_train)

#for validating model y_pred
y_pred= dc.predict(X_test)

#Actual Test y_pred
#y_pred= dc.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
cm=confusion_matrix(y_train,y_pred)
acc= accuracy_score(y_train, y_pred)

print('======================================')
print('======================================')
print(cm)
print('======================')
print(acc)


# # Visualising the Training set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Decision Tree Classification (Training set)')
# plt.xlabel('Factors')
# plt.ylabel('is_promoted')
# plt.legend()
# plt.show()
#
# # Visualising the Test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Decision Tree Classification (Test set)')
# plt.xlabel('Factors')
# plt.ylabel('is_promoted')
# plt.legend()
# plt.show()

