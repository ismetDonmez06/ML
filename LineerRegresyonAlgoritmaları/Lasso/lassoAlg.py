"""
Amaç hata kareler ortalamasını minimize eden katsayıları bu katsayılara ceza uygulayarak bulmaktır

Lasso regresyonu L1 yöntemi olarak geçer
Ridge regresyonuda L2 yöntemi olarak geçer

SSE l2,1= (i= 1 den n  kadar ) ∑ (yi-^yi)**2 + λ * (j= 1 den P  kadar ) ∑|Bj|**2

Ridge regresyonun ilgili ilgisiz tüm değişkenleri modelde bırakma dezavantajını gidermek
için önerilmiştir.
Lasso'da katsayıyı sıfıra yaklaştırır
Fakat L1 normu lambda yeteri kadar büyük oldugunda bazı katsayıları sıfır yapar Boylece değişken seçimi
yapmış olur.

Lambda nın doğru seçilmesi çok önemlidir buradada cv kullanılır

λ =Ayar paremetresi .Secilirken belirli değerler içiren bir küme seçilir ve cross val uygulanır ve
test hatası bulunur . En küçük değer atanır
B  =Bulunacak olan paremetreler(katsayılar)

"""
"""
işlem yaparken elimizde hem train hatası hemde test hatası olacak

train hatası test hatasının kotu bir tahmincisidir 
train hatasını kontrol ederek modlimizi iyileştirme yapabiliriz


"""
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV,LassoCV

df = pd.read_csv("Hitters.csv")
df =df.dropna()

#kategorik verileri dumy değişkenleri çevirme
dms = pd.get_dummies(df[['League', 'Division','NewLeague']])

y= df['Salary']
X_ = df.drop(['Salary','League','Division','NewLeague'] , axis=1).astype('float64')

X =pd.concat([X_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)
x_train, x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
print(df.shape)



lasso_model =Lasso(alpha=0.0001 ,normalize=False).fit(x_train,y_train)
#lasso_model =LassoCV().fit(x_train,y_train)

print(lasso_model.intercept_)
print(lasso_model.coef_)

#train hatası
y_pred =lasso_model.predict(x_train)
print(np.sqrt(mean_squared_error(y_train,y_pred)))

#test hatası
y_pred =lasso_model.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
print(r2_score(y_test,y_pred))














