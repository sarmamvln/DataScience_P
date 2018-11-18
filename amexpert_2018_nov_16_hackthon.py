import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
import csv

#target var -

trainpath= '\\train.csv'

historicpath='\\historical_user_logs.csv'

testpath= '\\test.csv'


trainset= pd.read_csv(trainpath, index_col='session_id')   #dropcolslis - ['DateTime']

historicset= pd.read_csv(historicpath)

testset= pd.read_csv(testpath, index_col='session_id')

commdropcolslis =['DateTime']

trainset.drop(labels=commdropcolslis, axis=1, inplace=True)

historicset.drop(labels=commdropcolslis, axis=1, inplace=True)

testset.drop(labels=commdropcolslis, axis=1, inplace=True)


# trainset.set_index('user_id')
# historicset.set_index('user_id')
# testset.set_index('user_id')



# print(trainset.head())
# print('=========================')
# print(historicset.head())
# print('=========================')
# print(trainset.info())
# print('=========================')
# print(historicset.info())
# print('=========================')

# trainset.join(historicset, on='user_id',how='left')
#
# print(trainset.head())
# print('====================')

# acttrain= trainset.join(newtrain, on='product', how='left')
#
# print(acttrain.head())

trainset.columns = [str.replace(' ', '_') for str in trainset.columns]
historicset.columns = [str.replace(' ', '_') for str in historicset.columns]
testset.columns= [str.replace(' ', '_') for str in testset.columns]

colslist= trainset.columns
targetcol= colslist[-1]


##Mising data percent by col code
# for col in trainset.columns:
#     print('Missing value percent for ',col, ' is = ', trainset[col].isnull().count()/len(trainset) )
# for col in testset.columns:
#     print('Missing value percent for ',col, ' is = ', testset[col].isnull().count()/len(testset) )


# #bar chart
# def plot_bars(dataset, cols):
#     for col in cols:
#         siz = plt.figure(figsize=(6,6))
#         axis = siz.gca()
#         counts = dataset[col].value_counts()
#         counts.plot.bar(ax = axis, color = 'blue')
#         axis.set_title('Number of  ' + col)
#         axis.set_xlabel(col)
#         axis.set_ylabel('Count of '+col)
#         plt.show()
#
# plot_bars(trainset, colslist)
#
# #Histograms
# def plot_histogram(dataset, cols, bins=10):
#     for col in cols:
#         size = plt.figure(figsize=(6, 6))  # define plot area
#         axis = size.gca()  # define axis
#         dataset[col].plot.hist(ax=axis, bins=bins)  # Use the plot.hist method on subset of the data frame
#         axis.set_title('Histogram of ' + col)  # Give the plot a main title
#         axis.set_xlabel(col)  # Set text for the x axis
#         axis.set_ylabel('Number of '+col)  # Set text for y axis
#         plt.show()
#
# plot_histogram(trainset, colslist)
#
# #ScatterPlot
# def plot_scatter(dataset, cols, col_y = targetcol):
#     for col in cols:
#         size = plt.figure(figsize=(7,6)) # define plot area
#         axis = size.gca() # define axis
#         dataset.plot.scatter(x = col, y = col_y, ax = axis)
#         axis.set_title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
#         axis.set_xlabel(col) # Set text for the x axis
#         axis.set_ylabel(col_y)# Set text for y axis
#         plt.show()
#
# plot_scatter(trainset, colslist)
#
#
# #Box whisker plot - relation between categorical and numeric
# def plot_box(dataset, cols, col_y=targetcol):
#     for col in cols:
#         sb.set_style("whitegrid")
#         sb.boxplot(col, col_y, data=dataset)
#         plt.xlabel(col)  # Set text for the x axis
#         plt.ylabel(col_y)  # Set text for y axis
#         plt.show()
#
# plot_box(trainset, colslist)


#Clean data function
def clean_process(dataset):
    objcols = []
    numcols = []

    # getting numcols and objcols
    for col in dataset.columns:
        if dataset[col].dtypes == 'object':
            objcols.append(col)
        elif dataset[col].dtypes in ['int32', 'float32', 'int64', 'float64']:
            numcols.append(col)
    #dataset.dropna(axis=0,how='any',inplace=True)

    dataset[objcols] = dataset[objcols].astype('category')
    dataset['prod'] = dataset['product'].cat.codes
    if 'gender' in objcols:
        dataset['gen'] = dataset['gender'].cat.codes

    elif  'action' in objcols:
        dataset['act'] =  dataset['action'].cat.codes

          

    dataset= dataset.drop(axis=1, columns=objcols)

    #dataset.fillna(0, inplace=True)
    for col in numcols:
        dataset[col] = dataset[col].fillna(dataset[col].mean())


    return dataset


dataset_train= clean_process(trainset)
dataset_hist= clean_process(historicset)
dataset_test= clean_process(testset)

# ##train test plit groups
def split_bygroups(df, col, test_size):
    from sklearn.model_selection import train_test_split
    train, test = pd.DataFrame(), pd.DataFrame()
    groby = df.groupby(col, as_index=True, sort=True)
    for k in groby.groups:
        traink, testk = train_test_split(groby.get_group(k), test_size=test_size,random_state=0)
        train = pd.concat([train, traink])
        test = pd.concat([test, testk])


    return train, test

train, test = split_bygroups(dataset_train, 'prod', test_size=0.30)

train_hist, test_hist= split_bygroups(dataset_hist, 'prod', test_size=0.30)
#dataset_hist.groupby('prod', as_index=True, sort=True)

dataset_test.groupby('prod', as_index=True, sort=True)





X_train= train.iloc[: ,:-1].values
y_train= train.iloc[:,-1].values

X_test= test.iloc[:, :-1].values
y_test= test.iloc[:, -1].values

#X_train_hist=dataset_hist.iloc[:, :].values
X_train_hist= train_hist.iloc[: ,:-1].values
y_train_hist= train_hist.iloc[:,-1].values

X_test_hist= test_hist.iloc[:, :-1].values
y_test_hist= test_hist.iloc[:, -1].values


X_act= dataset_test.iloc[:, :].values

# ##Outliner remove
# def replace_outliers(df, outlier_col=colslist[3]):
#     for col in outlier_col:
#         limit = df[col].quantile(0.99)
#         df[col] = df[col].mask(df[col] > limit, limit)
#     return df
#
# traindataset= replace_outliers(dataset)
# #testdataset1= replace_outliers(testdataset)



# # # # Applying Kernel PCA
from sklearn.decomposition import PCA
# pca = PCA()
# #pca= KernelPCA(n_components=)
# pca_comps = pca.fit(X_train)
# # X_train = pca.fit_transform(X_train)
# # X_test = pca.transform(X_test)
# explained_variance = pca.explained_variance_ratio_
#
# temp_explained_variance=explained_variance
# sorted(temp_explained_variance)
#
# print(sorted(temp_explained_variance))
# print('==============================')
# print(np.sum(pca_comps.explained_variance_ratio_))
# print('==============================')
# print(explained_variance.size)
# print('==============================')
#
# def plot_explained(mod):
#     comps = mod.explained_variance_ratio_
#     x = range(len(comps))
#     x = [y + 1 for y in x]
#     plt.plot(x,comps)
#     plt.show()
# plot_explained(pca_comps)

pca_mod_3 = PCA(n_components = 3)
X_train= pca_mod_3.fit_transform(X_train)
X_test= pca_mod_3.transform(X_test)


X_test_act= pca_mod_3.transform(X_act)



# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0,splitter='best')

#ind=classifier.apply(X_act['session_id'])

#classifier.fit(X_train_hist, y_train_hist)

classifier.fit(X_train, y_train)


# Predicting the Test set results
#y_pred_hist= classifier.predict(X_test_hist ,check_input=True)
y_pred = classifier.predict(X_test,check_input=True)


y_act_pred= classifier.predict(X_test_act, check_input=True)


##trainset pred Results
with open('\\1.csv', 'w') as res:
    wr = csv.writer(res)
    wr.writerow(y_act_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc= accuracy_score(y_test, y_pred)
# print(cm)
#
# print('--------------------')

print(acc)






