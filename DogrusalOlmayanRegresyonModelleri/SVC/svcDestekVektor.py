
"""
Destek vektor regresyonu SVC

Sınıflandırma ve regresyon için kullanılır
Robust(dayanıklı ) bir modeldir. Aykırı değerlere

Amaç bir marjin aralığını max noktayı en küçük hata ile alabilecek şekilde doğru yada eğri belirlemektir.

y =b0 +wx +e
(w=b1 demek)

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
from sklearn.tree import DecisionTreeClassifier
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

svr_model =SVR("linear").fit(x_train,y_train)
print(svr_model.intercept_)
print(svr_model.coef_)
y_pred = svr_model.predict(x_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))


#model tuning en iyi c ceza katsayısı değerini bulcaz

svr_params= {"C":[0.1,0.2,0.5,1,3,4]}

svr_cv_model =GridSearchCV(svr_model,svr_params,cv=4 ,n_jobs=-1).fit(x_train,y_train)
#verbose=2 parametresi eklersek çalışma esnasında durumu raporlayarak işlemi gerçekleştiri
#njobs = -1 yaparsak işlemci gücünü max yapar

print(svr_cv_model.best_params_)

#final model

svr_tuned =SVR("linear",C=0.5).fit(x_train,y_train)
y_pred = svr_tuned.predict(x_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))


