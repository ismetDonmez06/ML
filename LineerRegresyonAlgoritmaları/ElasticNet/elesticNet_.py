"""
ElasticNet L1 ve L2 yaklaşımını birleştirir.


SSE ElesticNet= (i= 1 den n  kadar ) ∑ (yi-^yi)**2 + λ1 * (j= 1 den P  kadar ) ∑Bj**2  + λ * (j= 1 den P  kadar ) ∑|Bj|**2

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split
import matplotlib as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import RidgeCV,LassoCV,ElasticNetCV

df = pd.read_csv("Hitters.csv")
df =df.dropna()
dms = pd.get_dummies(df[['League', 'Division','NewLeague']])

y= df['Salary']
X_ = df.drop(['Salary','League','Division','NewLeague'] , axis=1).astype('float64')

X =pd.concat([X_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)
x_train, x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
print(df.shape)


enet_model = ElasticNet(alpha=20,normalize=True).fit(x_train,y_train)
#enet_model = ElasticNetCV().fit(x_train,y_train)

y_pred = enet_model.predict(x_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))

print(r2_score(y_test,y_pred))