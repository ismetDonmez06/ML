import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale,StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,mean_squared_error,r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("diabetes.csv")

y= df["Outcome"]

x= df.drop(["Outcome"],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)



scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)

scaler.fit(x_test)
x_test=scaler.transform(x_test)

mlpc=MLPClassifier().fit(x_train,y_train)

y_preda =mlpc.predict(x_test)
print(accuracy_score(y_test,y_preda))



#model tuninig

mlcp_params = {"alpha" : [1,2,3,5,0.1,0.3],"hidden_layer_sizes":[(10,10),(100,100),(3,5)]}


newModel = MLPClassifier("lbfgs")

s=GridSearchCV(newModel,mlcp_params,cv=10,verbose=2).fit(x_train,y_train)
print(s.best_params_)


#son model

mlp = MLPClassifier(solver="lbfgs",alpha=1,hidden_layer_sizes=(3,5)).fit(x_train,y_train)

y_pred = mlp.predict(x_test)
print(accuracy_score(y_test,y_pred))







