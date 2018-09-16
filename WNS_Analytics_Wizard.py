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
traindataset= pd.read_csv(trainpath, header=None, names=colsname, index_col='employee_id',na_values='?')  #

#Actual test data read
testdataset= pd.read_csv(testpath, header=None, names=colsname[0:13], index_col='employee_id',na_values='?') #



objcols= ['department', 'region', 'education', 'gender', 'recruitment_channel']
numcols= ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'KPIs_met', 'awards_won', 'avg_training_score']


# for col in traindataset.columns:
#     if traindataset[col].dtype == object:
#         count = 0
#         count = [count + 1 for x in traindataset[col] if x == '?']
#         print(col + ' ' + str(sum(count)))



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

    dataset.fillna(0, inplace=True)
    # for col in numcols:
    #     dataset[col] = dataset[col].fillna(dataset[col].mean())


    return dataset



dataset= clean_process(traindataset)



#
# #Actual Testdata cleaning
dataset_tes= clean_process(testdataset)
#
#
# # #Bar chart
# # def plot_bars(dataset, cols):
# #     for col in cols:
# #         fig = plt.figure(figsize=(6,6))
# #         ax = fig.gca()
# #         counts = dataset[col].value_counts()
# #         counts.plot.bar(ax = ax, color = 'blue')
# #         ax.set_title('Number of Prometed by ' + col)
# #         ax.set_xlabel(col)
# #         ax.set_ylabel('Count of Promoted')
# #         plt.show()
# #
# # plot_bars(dataset, num_cols)
#
#


# # print(dataset.dtypes)
# # print('================================================')
# #Actual Test data dtypes
# #print(dataset_tes.dtypes)
#
#
X= dataset.loc[:, ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'KPIs_met', 'awards_won', 'avg_training_score','dum_department','dum_region','dum_education','dum_gender','dum_recruitment_channel']].values
y= dataset.loc[:, 'is_promoted'].values

#Actual Test data Provided
from sklearn.datasets.samples_generator import make_blobs
X_test_prov, y_pred_for_testset = make_blobs(n_features=12, random_state=1)
#X_test_prov= dataset.iloc[:].values


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

#Actual Test y_pred based on provide Test set
#y_pred_for_testset= dc.predict(X_test_prov)
y_pred_for_testset=dc.predict(X_test_prov)

from sklearn.metrics import accuracy_score,confusion_matrix
cm=confusion_matrix(y_test,y_pred)
acc= accuracy_score(y_test, y_pred)

print('======================================')
print('Model Confusion Matrix: ')
print('---------------------------')
print(cm)
print('======================')
print('Model Accuracy is : ', acc*100 ,'%')


print('*************************************************************')
print('Actual Test set provided y values: ')
for i in range(len(X_test_prov)):
	print("X=%s, Predicted=%s" % (X_test_prov[i], y_pred_for_testset[i]))

