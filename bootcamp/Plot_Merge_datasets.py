import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
fi1=pd.read_csv('items.csv')
fs1=pd.read_excel('sales_2023.xlsx',engine='openpyxl') #no need to use engine="" in linux
fs2=pd.read_excel('sales_2024.xlsx',engine='openpyxl')
mf=pd.merge(fi1,fs1,left_on='id',right_on='tid')
mf2=pd.merge(mf,fs2,left_on='id',right_on='tid')
gpo=mf2.groupby('name')
rf1=gpo.sum(numeric_only='True')
rf2=rf1.reset_index()
ff=rf2[['name','sale_qty_x','sale_qty_y']] #since 2023 sale_qty got renamed as sale_qty_x after merging
ff.plot(kind='bar',x='name',y=['sale_qty_x','sale_qty_y'],stacked='True')
plt.show()