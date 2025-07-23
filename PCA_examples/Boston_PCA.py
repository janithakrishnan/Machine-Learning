from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
bc=load_boston()
pca=PCA(3)
X=bc.data
y=bc.target
Xt=pca.fit_transform(X)
print(X.shape)
print(Xt.shape)
print(Xt)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
Xt_train,Xt_test,yt_train,yt_test = train_test_split(Xt,y,test_size = 0.2)
dt1=SVR()
dt2=SVR()
dt1.fit(X_train,y_train)
dt2.fit(Xt_train,yt_train)
p1=dt1.predict(X_test)
p2=dt2.predict(Xt_test)

from sklearn.metrics import confusion_matrix,accuracy_score,mean_squared_error
print("\nWithout using PCA")
print("RMSE:",pow(mean_squared_error(y_test,p1),-2))
print("\nAfter using PCA")
print("RMSE:",pow(mean_squared_error(yt_test,p2),-2))
