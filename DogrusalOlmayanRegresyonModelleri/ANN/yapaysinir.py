"""
insan beyin sinirlerini referasn alan sınıflandırma ve regresyon problemleri için kullanılan algoritmadır

Amaç en küçük hata ile tahmin yapacak kat sayıları bulma


Girdiler = x1 x2 x3 bağımsız değişkenin değerleri
Ağırlıklar = w1 w2 w3 katsayılar

girdiler ve ağırlıklar çarpıp toplanır
x1w1 + x2w2 +x3w3 değerler aktivasyon fonksiyonunda işlenip çktı   q değerleri elde edilir.
Bu çıktı diğer ağ elemanlarına yada diğer hücrelere aktarılıyor olacak.


input ----------------hidden-------------output  yapay sinir ağı

ara katman = girişten gelen bilgileri işler ve çıkışa götürür . ara kataman birden fazla katamn içerebilir.

ağın öğrenmesi demek ağırlıkların sürekli değiştirilerek kabul edilebilir hata miktarı elde edileceğine kadar devam eder.


Yapay sinir ağı gerçekleşme basamakları

1 örnek veri seti toplanır
2 ağın topolajik yapısına kara verilir
3 ağdaki ağırlıkları başlangıç değerleri atanır.
4 örnek veri seti ağa sunulur
5 ileri hesaplama işlemleri gerçekleşir.
6 gerçek çıktılar ile tahmin çıktıları karşılaştırılır
7 ağırlıklar günclenir ve öğrenme tamamlanır


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


#standırtlaşma işlemi gerçekleştircez
#homojen verilerde daha iyi çalıştığı için yapıyoruz
#aykırı değerlerde çok iyi çalışmamaktadır.

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled =scaler.transform(x_train)
scaler.fit(x_test)
x_test_scaled =scaler.transform(x_test)

mlp_model = MLPRegressor().fit(x_train_scaled,y_train)
y_pred = mlp_model.predict(x_test_scaled)
print(np.sqrt(mean_squared_error(y_test,y_pred)))

#model tuning
mlp_params = {"alpha": [0.1,0.01,0.001,0.02],
              "hidden_layer_sizes": [(10,2),(100,100),(7,8),(5,5)]}
#hidden layer de 2 katman oluşturuyor ve 10 nören ve diğer katman 2 nörön alıyor
#istediğimiz kadar katman olabilir örn 3 katmanlı
#(20,8,7) gibi
mlp_cv_model = GridSearchCV(mlp_model,mlp_params,cv=3,verbose=2,n_jobs=-1).fit(x_train_scaled,y_train)
print(mlp_cv_model.best_params_)

#final model

model = MLPRegressor(alpha=0.02,hidden_layer_sizes=(100,100)).fit(x_train_scaled,y_train)
y_pred = model.predict(x_test_scaled)
print(np.sqrt(mean_squared_error(y_test,y_pred)))