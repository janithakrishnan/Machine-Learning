from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#READING DATA FROM CSV
mat_data=pd.read_csv('Maternal Health Risk Data Set.csv')

#CHANGING PD DATAFRAME TO NP ARRAY
mat=mat_data.to_numpy()
#EXTRACTING FEATURES
X=mat[:,:-1]
#EXTRACTING TARGET 
y=mat[:,-1]
#CHANGING STRING VALUES IN y TO INTEGERS
unique_labels,integer_labels=np.unique(y,return_inverse=True)

#PRINTING LABELS
print("Unique Labels (Target):",unique_labels)
print("Integer Labels (Target):",integer_labels)

#COPYING INTEGER LABELS TO ANOTHER NP ARRAY yy (not necessary)
yy=integer_labels.copy()
print(X.shape)
print(yy.shape)

##KNN CLASSIFICATION
knn=KNeighborsClassifier()
knn.fit(X,yy)
p=knn.predict(X)

#PRINTING METRICS
print("CONFUSION MATRIX\n",confusion_matrix(yy,p))
print("ACCURACY SCORE\n",accuracy_score(yy,p))
