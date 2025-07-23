from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
bc=load_breast_cancer()
X=bc.data
y=bc.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

dt1=LinearDiscriminantAnalysis()
dt1.fit(X_train,y_train)
p1=dt1.predict(X_test)


from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,p1))
print(accuracy_score(y_test,p1))

