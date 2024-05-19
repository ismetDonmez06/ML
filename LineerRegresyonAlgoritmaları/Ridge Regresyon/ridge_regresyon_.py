"""
Amaç hata kareler toplamını minumize edecek katsayıları bu katsayıları bir ceza uygulayarak
bulmaktır.


SSE l2= (i= 1 den n  kadar ) ∑ (yi-^yi)**2 + λ * (j= 1 den P  kadar ) ∑Bj**2

λ =Ayar paremetresi .Secilirken belirli değerler içiren bir küme seçilir ve cross val uygulanır ve
test hatası bulunur . En küçük değer atanır
B  =Bulunacak olan paremetreler(katsayılar)




#aşırı öğrenmye karşı dirençli
# yanlıdır fakat bvaryansı düşüktür
#çok boyutluluk karşı çözüm sunar ( değişken sayısının gözlem sayısında fazla olmasıdır)
# çoklu doğrusal bağlantı problemi oldupunda etkilidir(bağımsız değişkenler arasında yüksek
corelasyon olmasıdır)





"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV

df = pd.read_csv("Hitters.csv")
df =df.dropna()
dms = pd.get_dummies(df[['League', 'Division','NewLeague']])

y= df['Salary']
X_ = df.drop(['Salary','League','Division','NewLeague'] , axis=1).astype('float64')

X =pd.concat([X_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)
x_train, x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
print(df.shape)

ridge_model = Ridge(alpha=0.1).fit(x_train,y_train)

lambdalar = 10**np.linspace(10,-2,100)*0.5

rid_model =Ridge()
katsayilar = []
for i in lambdalar:
    rid_model.set_params(alpha=i)
    ridge_model.fit(x_train,y_train)
    katsayilar.append(ridge_model.coef_)


# tahmin
y_pred = ridge_model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)

from sklearn.model_selection import cross_val_score
#cross validation

a=np.sqrt(np.mean(-cross_val_score(ridge_model,x_test,y_test,cv=10,scoring="neg_mean_squared_error")))
print(a)

#model tuning

lambdalar1 = np.random.randint(0,1000,100)

lambdalar2 = 10**np.linspace(10,-2,100)*0.5


ridgecv=RidgeCV(alphas=lambdalar2,scoring="neg_mean_squared_error",cv=10,normalize=True)

ridgecv.fit(x_train,y_train)

print(ridgecv.alpha_)

#final modeli
ridgeFinal=Ridge(alpha=ridgecv.alpha_).fit(x_train,y_train)


y_pred = ridgeFinal.predict(x_test)
a=np.sqrt(mean_squared_error(y_test,y_pred))
print(a)