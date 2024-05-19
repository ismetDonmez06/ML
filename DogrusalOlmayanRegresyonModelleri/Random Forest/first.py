"""
Topluluk öğrenme yaklaşımıdır.
Birden fazla algoritmanın yada bir den falza ağacın
bir araya gelerek toplu bir şekilde öğrenmesi yada tahmin etmeye
çalışmasıdır.

Temeli boostrap yöntemi ile oluşturan birden fazla karar ağacının
ürettiği tahminlerin bir araya getilerek değerlendirilmesine
dayanır

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



rf_model = RandomForestRegressor(random_state=42).fit(x_train,y_train)
y_pred =rf_model.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))


#model tuning

params = {"max_depth":range(1,3),"max_features" : [1,2,5],
          "n_estimators": [200,500,1000]}


model = GridSearchCV(rf_model,params,cv=3).fit(x_train,y_train)
print(model.best_params_)

#final model

rf_model = RandomForestRegressor(random_state=42,max_depth=2,max_features=5).fit(x_train,y_train)
y_pred =rf_model.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))


##DEĞİSKEN ÖNEM SIRALAMASI

Importance = pd.DataFrame({"Importance":rf_model.feature_importances_*100},index=x_train.columns)


Importance.sort_values(by="Importance",axis=0,ascending=True).plot(kind='barh',color='r')

plt.xlabel("variable importance")
plt.gca().legend_=None
plt.show()

