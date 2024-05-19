"""
Adaboost'un sınıflandırma ve regresyon problemlerine uyarlanabilen genelleştirilmiş
versiyondur.

Artıklar (gerçek değerler - tahmin edilen)üzerine tek bir tahminsel model formunda olan modeller
serisi kurulur.

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

from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("D:\PythonProjects\DogrusalOlmayanRegresyonModelleri\KEnYakınKomşuAlgoritması\Hitters.csv")
df =df.dropna()
dms = pd.get_dummies(df[['League', 'Division','NewLeague']])

y= df['Salary']
X_ = df.drop(['Salary','League','Division','NewLeague'] , axis=1).astype('float64')

X =pd.concat([X_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)
x_train, x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
print(x_train)



gbm_model = GradientBoostingRegressor().fit(x_train,y_train)
y_pred =gbm_model.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))

#model tuning

# rabast yontemler (dayanıklı yontemler) artıklara
#karşı duyarlı  artık:(gerçek değer -tahmin edilen deger)
gbmParams={"learning_rate" : [0.1,0.01,0.002],
           "max_depth": [3,5,8],
           "n_estimators" :[100,200,300]
        }

gbmModel=GradientBoostingRegressor().fit(x_train,y_train)
gmb_cv = GridSearchCV(gbmModel,gbmParams,cv=2).fit(x_train,y_train)

print(gmb_cv.best_params_)


#final model

gbm_model = GradientBoostingRegressor(learning_rate=0.1,max_depth=3,n_estimators=100).fit(x_train,y_train)
y_pred=gbm_model.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))

