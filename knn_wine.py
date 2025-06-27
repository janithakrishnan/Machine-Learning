from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
wine=load_wine()
X=wine.data
Y=wine.target
#Choosing the classifier
knn=KNeighborsClassifier(5)
#Creating the model
knn.fit(X,Y)
#Using the model to predict X
p=knn.predict(X)
print("ACCURACY\n",accuracy_score(Y,p))
print("CONFUSION MATRIX\n",confusion_matrix(Y,p))
print("CLASSIFICATION REPORT\n",classification_report(Y,p))
