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
print(df.head())

print(df["Outcome"].value_counts())

print(df.describe().T)

y= df["Outcome"]

x= df.drop(["Outcome"],axis=1)


loj_model = LogisticRegression(solver="liblinear").fit(x,y)
print(loj_model.intercept_)
print(loj_model.coef_)

y_pred = loj_model.predict(x)
print(confusion_matrix(y,y_pred))
print(accuracy_score(y,y_pred))


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


loj_Cv_model = LogisticRegression(solver="liblinear").fit(x_train,y_train)
y_pred = loj_Cv_model.predict(x_test)

print(accuracy_score(y_test,y_pred))


print(cross_val_score(loj_Cv_model,x_test,y_test,cv=10))






