import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn

trainpath= '\\WNS Analytics Wizard  2018_train_LZdllcl.csv'
testpath= '\\WNS Analytics Wizard  2018_test_2umaH9m.csv'

colsname=['department', 'region', 'education', 'gender', 'recruitment_channel',  'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'KPIs_met', 'awards_won', 'avg_training_score', 'is_promoted']
traindataset= pd.read_csv(trainpath, index_col='employee_id')
traindataset.columns = [str.replace('?', '') for str in traindataset.columns]



trainobjcols= ['department', 'region', 'education', 'gender', 'recruitment_channel']

print(traindataset.head)
print('============================')
print(traindataset.dtypes)
print('============================')




print('================================================')

for col in trainobjcols:
    print(col, 'Unique values count: \t', traindataset.count().nunique())


X= traindataset.iloc[:, :-1].values
y= traindataset.iloc[:, -1].values

