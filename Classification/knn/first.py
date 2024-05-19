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

knn_model = KNeighborsClassifier().fit(x_train,y_train)

y_pred = knn_model.predict(x_test)

print(accuracy_score(y_test,y_pred))



#model tunining

knn =KNeighborsClassifier()
knn_params = {"n_neighbors" :np.arange(1,50)}

knn_cv_model = GridSearchCV(knn,knn_params,cv=10).fit(x_train,y_train)
print(knn_cv_model.best_params_)


knn_tuned = KNeighborsClassifier(n_neighbors=11).fit(x_train,y_train)
y_pred = knn_tuned.predict(x_test)
print(accuracy_score(y_test,y_pred))

