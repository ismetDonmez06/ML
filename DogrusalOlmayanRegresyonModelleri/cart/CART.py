"""
Regresyon ve classification için kullanılır

Amaç veri seti içerisindeki karmaşık yapıları basit karar
yapılarına dönüştürmektir

heterojen veri setleri belirlenmiş bir hedef değişkene göre
homojen alt gruplara ayrılır.

aşırı öğrenmeye meyillidir.

"""


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor,LocalOutlierFactor,NeighborhoodComponentsAnalysis,KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import  neighbors
from sklearn.svm import SVR
from warnings import filterwarnings
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from scipy.stats import norm, skew, describe
import xgboost as xgb

df = pd.read_csv("D:\PythonProjects\DogrusalOlmayanRegresyonModelleri\KEnYakınKomşuAlgoritması\Hitters.csv")
df =df.dropna()
dms = pd.get_dummies(df[['League', 'Division','NewLeague']])

y= df['Salary']
X_ = df.drop(['Salary','League','Division','NewLeague'] , axis=1).astype('float64')

X =pd.concat([X_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)
x_train, x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
print(x_train)


#tek değişkenli atış değeri ile maaş arasındaki ilişki açıklıcaz
x_train = pd.DataFrame(x_train["Hits"])
x_test = pd.DataFrame(x_test["Hits"])

cart_model = DecisionTreeRegressor()
cart_model =cart_model.fit(x_train,y_train)

y_pred = cart_model.predict(x_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))


#birden fazla değişkenli


df = pd.read_csv("D:\PythonProjects\DogrusalOlmayanRegresyonModelleri\KEnYakınKomşuAlgoritması\Hitters.csv")
df =df.dropna()
dms = pd.get_dummies(df[['League', 'Division','NewLeague']])

y= df['Salary']
X_ = df.drop(['Salary','League','Division','NewLeague'] , axis=1).astype('float64')

X =pd.concat([X_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)
x_train, x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
print(x_train)


cart_model = DecisionTreeRegressor()
cart_model =cart_model.fit(x_train,y_train)
y_pred = cart_model.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))


#model tuning
#hyper paremetreler

#max_dept kaç tane dallanma olacak
params =  {"max_depth" : [2,3,4,5,10,20],"min_samples_split":[2.,10,5,7]}

cart_model = DecisionTreeRegressor()
carr_cv_model = GridSearchCV(cart_model,params,cv=5).fit(x_train,y_train)

print(carr_cv_model.best_params_)
 
#final model

cart_model = DecisionTreeRegressor(max_depth=3,min_samples_split=5).fit(x_train,y_train)

y_pred=cart_model.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
