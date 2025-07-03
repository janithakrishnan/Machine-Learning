from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import pandas as pd
immuno_data=pd.read_csv('/home/sw900b4_janitha/Assignments Janitha/MACHINE LEARNING/assignments and notes/ML_assignments_janitha/4/Immunotherapy.csv')
immuno=immuno_data.to_numpy()
X=immuno[:,:-1]
y=immuno[:,-1]
print(X.shape)
print(y.shape)
knn3=KNeighborsClassifier(3)
knn3.fit(X,y)
p3=knn3.predict(X)
knn5=KNeighborsClassifier(5)
knn5.fit(X,y)
p5=knn5.predict(X)
knn7=KNeighborsClassifier(7)
knn7.fit(X,y)
p7=knn7.predict(X)
knn9=KNeighborsClassifier(9)
knn9.fit(X,y)
p9=knn9.predict(X)
knn11=KNeighborsClassifier(11)
knn11.fit(X,y)
p11=knn11.predict(X)


print("ACCURACY for k=3\n",accuracy_score(y,p3))
print("ACCURACY for k=5\n",accuracy_score(y,p5))
print("ACCURACY for k=7\n",accuracy_score(y,p7))
print("ACCURACY for k=9\n",accuracy_score(y,p9))
print("ACCURACY for k=11\n",accuracy_score(y,p11))

import matplotlib.pyplot as plt
plt.xlabel("k VALUES")
plt.ylabel("ACCURACY")
plt.title("IMMUNOTHERAPY-KNN CLASSIFICATION")
k=[3,5,7,9,11]
a=[accuracy_score(y,p3),accuracy_score(y,p5),accuracy_score(y,p7),accuracy_score(y,p9),accuracy_score(y,p11)]
plt.xticks(k,labels=k)
plt.bar(k,a)
plt.show()
