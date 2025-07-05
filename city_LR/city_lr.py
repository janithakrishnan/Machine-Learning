#LINEAR REGRESSION ON city.csv DATASET
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
city=pd.read_csv('city.csv')
city=city.to_numpy()
X=city[:,1:-1]
y=city[:,-1]
print(X.shape)
print(y.shape)

lr=LinearRegression()
lr.fit(X,y)
p=lr.predict(X)
print("R2_Score:",r2_score(y,p))

import matplotlib.pyplot as plt
plt.xlabel('ACTUAL VALUES')
plt.ylabel('PREDICTED VALUES')
plt.title('PREDICTION ON: Average parking rates per month')
plt.scatter(y,p)
plt.show()