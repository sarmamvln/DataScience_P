import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy
import pandas.util

path='\\insurance.csv'

data= pd.read_csv(path)

# print(data.info())
# print('===============================================')
# print(data.head(5))
# print('===============================================')
# print(data.describe(include='all'))
# print('===============================================')
# print(data.isnull().sum())
# print('===============================================')

data1= pd.get_dummies(data, prefix='dum_', columns= ['sex', 'smoker', 'region'], drop_first=True)

# print(data1['dum__male'].head(1))
# print(data1['dum__yes'].head(1))
# print(data1['dum__northwest'].head(1))
# print(data1['dum__southeast'].head(1))
# print(data1['dum__southwest'].head(1))
#data1.drop(['sex', 'smoker', 'region'], axis=1)

# print(data1.head())
# print('===============================================')
# print(data1.describe(include='all'))
# print('===============================================')
# print(data1.isnull().sum())
# print('===============================================')


X= data1.iloc[:, [0, 1, 2, 4,5, 6, 7, 8]].values
y= data1.iloc[:, 3].values

print(data1.dtypes)
print('===========================')

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, train_size=0.75, random_state=0)

# from sklearn.preprocessing import StandardScaler
# sc= StandardScaler
# X_train=sc.fit_transform(X_train)
# X_test= sc.fit_transform(X_test)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train, y_train)

y_pred= lr.predict(X_test)

from sklearn.metrics import regression
R_square= regression.r2_score(y_test, y_pred)

#print(R_square*100)


# plt.scatter(X_test, y_test, color='red')
# plt.plot(X_test, regression.predict(X_test), color='green')
# plt.title('InsuranceValue Vs Diagnosis (Test dataset)')
# plt.xlabel('Diagnosis')
# plt.ylabel('InsuranceValue')
# plt.show()


#New Dataset Prediction for Future
X_new= {}

#print(type  (X_new))
X_new['age']= int(input("Enter Age of Person : "))
X_new['sex']= input("Enter sex of Person( Male / Female ): ").lower()
X_new['bmi']= float(input("Enter bmi of Person: "))
X_new['children']= int(input("Enter number of Childern: "))
X_new['smoker']= input("Enter smoking status of Person( Yes / No ): ").lower()
X_new['region']= input("Enter region of Person (southeast / southwest / northeast / northwest): ").lower()



new_set= pd.DataFrame([X_new],columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

print(new_set)
print('============================================')
#new_set= pd.DataFrame.from_dict(X_new.values(), orient='columns', columns=['age', 'sex', 'bmi', 'smoker', 'region'])
#print(new_set)
#print(new_set.shape)

X_new_dum={}
X_new_dum['dum__male']= new_set.sex.map({'male':1, 'female':0})
X_new_dum['dum__yes']= new_set.smoker.map({'yes':1, 'no':0})
X_new_dum['dum__northwest'] = new_set.region.map({'northwest':0, 'northeast':1})
X_new_dum['dum__southeast'] = new_set.region.map({'southeast':0})
X_new_dum['dum__southwest'] = new_set.region.map({'southwest':1})


#temp= pd.DataFrame([X_new_dum],columns=['dum__male', 'dum__yes', 'dum__northwest', 'dum__southeast', 'dum__southwest'])  #,dtype='uint8'

new_set= new_set.append(X_new_dum, ignore_index=True)


new_set.replace(np.nan, 0, inplace=True)

#DTYPE CONVERSION
new_set.drop(columns=['sex', 'smoker', 'region'], inplace=True)
new_set['bmi']=new_set['bmi'].infer_objects()
new_set[['age', 'children']]=new_set[['age', 'children']].astype(int)
new_set[['dum__male', 'dum__yes', 'dum__northwest', 'dum__southeast', 'dum__southwest']]= new_set[['dum__male', 'dum__yes', 'dum__northwest', 'dum__southeast', 'dum__southwest']].astype('uint8')

print(new_set.dtypes)
out=lr.predict(new_set)
print("Predicted Insurance value for the Person based on Input: ",  out )

