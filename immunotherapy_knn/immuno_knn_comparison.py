from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import pandas as pd
immuno_data=pd.read_csv('Immunotherapy.csv')
immuno=immuno_data.to_numpy()
X=immuno[:,:-1]
y=immuno[:,-1]
accuracy=[]
k=[]
for i in range(1,16,2):
    knn=KNeighborsClassifier(i)
    knn.fit(X,y)
    p=knn.predict(X)
    a=accuracy_score(y,p)
    accuracy.append(a)
    k.append(i)
    print(f"ACCURACY for k={i}\n {accuracy_score(y,p)}")

import matplotlib.pyplot as plt
plt.xlabel("k VALUES")
plt.ylabel("ACCURACY")
plt.title("IMMUNOTHERAPY-KNN CLASSIFICATION")
plt.xticks(k,labels=k)
plt.plot(k,accuracy)
plt.show()