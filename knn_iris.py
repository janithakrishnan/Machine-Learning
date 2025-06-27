from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
iris=load_iris()
X=iris.data
Y=iris.target
knn=KNeighborsClassifier(5)
knn.fit(X,Y)
p=knn.predict(X)
print("ACTUAL TARGET\n",Y)
print("PREDICTED TARGET\n",p)
print("CONFUSION MATRIX\n",confusion_matrix(Y,p))
print("ACCURACY SCORE",accuracy_score(Y,p))
