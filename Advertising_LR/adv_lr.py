from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import pandas as pd
adv=pd.read_csv('Advertising.csv')
adv=adv.to_numpy()
X=adv[:,:-1]
y=adv[:,-1]
lr=LinearRegression()
lr.fit(X,y)
p=lr.predict(X)
print(f"ACCURACY: {accuracy_score(y,p)}")