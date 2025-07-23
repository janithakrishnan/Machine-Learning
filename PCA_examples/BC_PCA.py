from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
bc=load_breast_cancer()
pca=PCA(2)
X=bc.data
y=bc.target
Xt=pca.fit_transform(X)
print(X.shape)
print(Xt.shape)
print(Xt)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
Xt_train,Xt_test,yt_train,yt_test = train_test_split(Xt,y,test_size = 0.2)
dt1=SVC()
dt2=SVC()
dt1.fit(X_train,y_train)
dt2.fit(Xt_train,yt_train)
p1=dt1.predict(X_test)
p2=dt2.predict(Xt_test)

from sklearn.metrics import confusion_matrix,accuracy_score
print("\nWithout using PCA")
print(confusion_matrix(y_test,p1))
print(accuracy_score(y_test,p1))
print("\nAfter using PCA")
print(confusion_matrix(yt_test,p2))
print(accuracy_score(yt_test,p2))
