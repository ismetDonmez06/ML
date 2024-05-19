"""
Doğrusal olmayan regresyon modeli
Regresyon = bağımlı değikenin sayısal olduğu durumdaki makine öğrenmesi tahmin problemleri

K-Nearest Neighbors

Gözlemlerin birbireleri üzerinden benzerliklerine göre tahmin işlemi gerçekleştirilir.

Paremetrik olmayan bir öğrenme türüdür.

Büyük veri yapılarında çok başarılı değildir.


Bagımsız değişken değerleri verilen  gözlem biriminin bağımlı değişken birimi olan Y
sini tahmin etmek için diğer gözlem birimleriyle benzerliklerine bakılır

bu gözlem değerlerinin diğer bağımsız değişkenlerle uzaklığı hesaplanır

en yekın k adet gözlemin y değerleinin ortalamaısnı alıyoruz

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

df = pd.read_csv("Hitters.csv")
df =df.dropna()
dms = pd.get_dummies(df[['League', 'Division','NewLeague']])

y= df['Salary']
X_ = df.drop(['Salary','League','Division','NewLeague'] , axis=1).astype('float64')

X =pd.concat([X_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)
x_train, x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
print(x_train)


#model

model = KNeighborsRegressor().fit(x_train,y_train)
print(model.n_neighbors)
#bakabilceğimiz değerler paremetreler
print(dir(model))

y_pred = model.predict(x_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))

#model tuning
#hyper paremetreleri belirlemek içi
#GridSearchCV kullanılır

knn_pars={"n_neighbors": np.arange(1,30,1)}
knn = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn,knn_pars,cv=10).fit(x_train,y_train)
print(knn_cv_model.best_params_)

#final model

knn = KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"]).fit(x_train,y_train)
y_pred = knn.predict(x_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))












