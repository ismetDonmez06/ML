"""
Regresyon = Tahmin edilen değerin numeric yanı sayısal oluduğu durumlardır
Classification = Tahmin edilen değerin kategorik yani erkek kadın olur olmaz gibi olduğu durumdur

Supervisid (denetimli öğrenme) = tahmin edilen değerin veride oluduğu durumdur
y değişkenin veride olduğu durumdur.

Unspervisid(denetimsiz öğrenme ) = tahmin edilen değerin veri olmadıpı y
değişkenin veride olmadığı durumdur.

Basit lineer regresyon
yi = b0 + bi x

bo ve b1 katsayılarını veri içerisinden bulacağız yapay zeka kısmı dediği yer burası
b0 =y eksenini kestiği nokta
b1 =eğim

sse =(i= 1 den n  kadar ) ∑ (yi - ^yi)**2
sse = (i= 1 den n  kadar ) ∑ (yi - (b0+b1xi))**2

b0 =  (i= 1 den n  kadar ) ∑ (xi -xort) (yi-yort) / (i= 1 den n  kadar ) ∑ (xi-xort)**2

b1 = yort -b1xort


******************

rkare
bagımsız değişkenlerin bağımlı değişkeni açıklma yüzdesi
sadece regresyon modellerinde kullanılır

****************


Temel amaç bağımlı ve bağımsız değişkenler arasındaki doğrusal fonksiyonu bulma

yi = b0 + b1 Xi1 + b2 Xi2 + ..... +bp Xip +ei

yi = bagımlı değişken
b0 b1 ... = bagımsız değişkemlerin katsayıları model içerisindeki bağımsız değişkenlerin etkisini
kontrol etmek için kullanılır. Veri içerisinde bu katsayıları buluyoruz . Yapay zeka
kısmı dedikleri şey budur .

Bu katsayıları bulmak için

(i= 1 den n  kadar ) ∑ ei**2 = (i= 1 den n  kadar ) ∑ (yi -^yi)**2

b = (XTranspozy .X )**-1 XTranspozu .Y

bu işlemin sonucunda b değerlerin katsayılarını içerien array ortaya çıkacaktır


# Hatalar normal dağılır
# Hatalar birbirinde bagımsızdır ve aralarında otokorelasyon yoktur
# Her bir gözlem için hata terimleri varyansları sabittir
# Değişkenler ve hata terimleri arasında ilişki yoktur.
# Bagımsız değişkenler arasında çoklu doğrusal ilişki problemi yoktur.


"""

import pandas as pd
from sklearn.metrics import mean_squared_error

df = pd.read_csv("Advertising.csv")
df =df.iloc[:,1:len(df)]
print(df.head())

X= df.drop('sales' ,axis=1)
Y =df["sales"]

print(X.head())
print(Y.head())

#scikit learn ile model kurmak
from sklearn.linear_model import LinearRegression
lm =LinearRegression().fit(X,Y)

print(lm.intercept_)
print(lm.coef_)

yeni_veri =[[10],[30],[80]]

pred =lm.predict(X)

mse = mean_squared_error(Y,pred)

import numpy as np
rmse = np.sqrt(mse)

#model tuning (model doğrulama)

from sklearn.model_selection import train_test_split

x_train ,x_test,y_train,y_test =train_test_split(X,Y,test_size=0.3,random_state=99)

print(x_train.head())

model = LinearRegression().fit(x_train,y_train)
y_pred = model.predict(x_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))
from sklearn.model_selection import cross_val_score
#cross validation

a=np.sqrt(np.mean(-cross_val_score(model,x_test,y_test,cv=10,scoring="neg_mean_squared_error")))
print(a)




""" basit doğrusal regresyon
 1 tane bağımsız değişkenden oluşan yapılar
 Temel amaç bağımlı ve bağımsız değişkenler arasındaki ilişkiyi
 ifade edeb doğrusal fonksiyonu bulmaktır.

"""

import pandas as pd

df = pd.read_csv("D:\pythonisprojesi2\cccc.csv")
df = df.iloc[:,1:len(df)]
print(df.head())
print(df.info())

import seaborn as sns

from sklearn.linear_model import LinearRegression

X =df[["TV"]]
y = df[["sales"]]

reg = LinearRegression()
model = reg.fit(X,y)

print(model.intercept_) #b0
print(model.coef_) #b1

#r2 ifadesi modelin  bagımsız değişkenlerin bağımlı değişkeni açıklama
# oranı % şekilde açıklar

print(model.score(X,y))


##tahmin işlemi

#Sales = 7.03 +0.04*TV

print(model.predict([[165]]))

yeni_veri = [[5],[15],[30]]

print(model.predict(yeni_veri))
print("****************************")
#Artıklar (Hatalar)
#MSE hata kareler ortalaması ve RMSR karekoku

print(y.head(10)) #gerçek değerler
print(model.predict(X)[0:10]) #tahmin edilen dğerler

gercek_y= y[0:10]
tahmin_edilen =pd.DataFrame(model.predict(X)[0:10])


hatalar = pd.concat([gercek_y,tahmin_edilen],axis=1)
hatalar.columns =["gercek_y","tahmin edilen_y"]

print(hatalar)
#hata hesaplama
print("*******************")
hatalar["hata"] = hatalar["gercek_y"]-hatalar["tahmin edilen_y"]
print(hatalar)
#mse
hatalar["hata kareler"] =hatalar["hata"]**2
print(hatalar)
import numpy as np
#mse
print(np.mean(hatalar["hata kareler"]))

print("*************************")
print("*************************")
print("*************************")

""" çoklu doğrusal regresyon """

print("*************************")

df = pd.read_csv("D:\pythonisprojesi2\cccc.csv")
df = df.iloc[:,1:len(df)]
print(df.head())
print(df.info())

x = df.drop('sales',axis=1)
y= df[["sales"]]

##scitlearn

model=LinearRegression().fit(x,y)
print(model.intercept_)
print(model.coef_)

##tahmin
#sales = 2.94 + tv*0.04 + radio*0.19+ news*0.001

# 30 birim tv ,10 birim radida ,40 birim gazetede
#reklamında oluşacak satış nedir

yeniver = [[30],[10],[40]]

yeniver = pd.DataFrame(yeniver).T
print(yeniver)
print(model.predict(yeniver))
from sklearn.metrics import mean_squared_error
y_pred=model.predict(x)
#mse
print(mean_squared_error(y,y_pred))
#rmse
print(np.sqrt(mean_squared_error(y,y_pred)))

#model tuning

#sınama seti yaklaşımıya hata hesaplama
from sklearn.model_selection import  train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=99)

#train hatası
lm = LinearRegression().fit(x_train,y_train)
print(mean_squared_error(y_train,model.predict(x_train)))

#test hatası
lm = LinearRegression().fit(x_train, y_train)
print(mean_squared_error(y_test, model.predict(x_test)))

## k katlı cross validation ile hata hesaplama

from sklearn.model_selection import cross_val_score
#train icin
#cv mse
a=np.mean(-cross_val_score(model,x_train,y_train,cv=10,scoring="neg_mean_squared_error"))
print(a)
#cv resme
a=np.sqrt(np.mean(-cross_val_score(model,x_train,y_train,cv=10,scoring="neg_mean_squared_error")))
print(a)






