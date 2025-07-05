#LINEAR REGRESSION ON loan2.csv DATASET
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score
import pandas as pd
loan=pd.read_csv('loan2.csv')
loan=loan.to_numpy()
X=loan[:,1:-1]
y=loan[:,-1]

lr=LinearRegression()
lr.fit(X,y)
p=lr.predict(X)
print("Mean Absolute Error:",mean_absolute_error(y,p))
print("Mean Squared Error:",mean_squared_error(y,p))
print("Root Mean Squared Error:",root_mean_squared_error(y,p))
print("R2 Score:",r2_score(y,p))
