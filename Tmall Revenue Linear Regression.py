#!/usr/bin/env python
# coding: utf-8

# # Import Files

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression


# In[2]:


# check the current file path
import os
os.getcwd()


# In[3]:


# check if the file path exists
filepath = './Desktop/Tmall Integration_2.csv'
if not os.path.exists(filepath):
    print('file not exist!')


# In[4]:


df_tmall = pd.read_csv('./Desktop/Tmall Integration_only_receive.csv') #比较短的发货和gmvdata
#df_tmall = pd.read_csv('./Desktop/Tmall Integration_2.csv') #比较短的发货和gmvdata
#df_tmall = pd.read_csv('./Desktop/Tmall Integration_3.csv') #更久的gmvdata


# # View Data

# In[5]:


df_tmall.head()


# In[6]:


df_tmall.describe()


# In[7]:


#check data type
df_tmall.info()


# In[8]:


# change data type
df_tmall['date'] = pd.to_datetime(df_tmall['date'])


# In[9]:


df_tmall.head()


# # View Correlation through Scatter Plot

# In[10]:


import matplotlib.pyplot as plt #Matplotlib – Python画图工具库
import seaborn as sns #Seaborn – 统计学数据可视化工具库
# show scatter plot bettween multiple independent variables and the dependent variable
sns.pairplot(df_tmall, x_vars=['receive_pack_num_1d','receive_pack_num_today','receive_pack_num_today','receive_pack_num_today'], 
                          y_vars='tmall_revenue', 
                          height=4, aspect=1, kind='scatter')
plt.show() # show plot


# In[11]:


# Correlation Matrix
corr = df_tmall.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[12]:


# Correlation Matrix Plot
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.api as smg


#df_tmall_wo_date = df_tmall.drop(['date'], axis=1)

#corr_matrix = np.corrcoef(df_tmall_wo_date)
#smg.plot_corr(corr_matrix)
#plt.show()


# In[13]:


df_tmall.head()


# # Align X and y

# In[14]:


# Align X and y
#Create X

# X = df_tmall.drop(['tmall_revenue','date','sales_volume'], axis=1)

# File2 receive package ver
X = df_tmall[['receive_pack_num_9d', 'receive_pack_num_8d',
            'receive_pack_num_4d','receive_pack_num_today']] #'avg_price_per_merch_10d','oms_gmv_10d',

#backup-wording
# X = df_tmall[['receive_pack_num_10d','receive_pack_num_9d','receive_pack_num_8d','receive_pack_num_7d',
#             'receive_pack_num_6d','receive_pack_num_5d','receive_pack_num_4d','receive_pack_num_3d',
#            'receive_pack_num_2d','receive_pack_num_1d','receive_pack_num_today']]

# File1 gmv ver
# X = df_tmall[['oms_gmv_10d','oms_gmv_9d','oms_gmv_8d','oms_gmv_7d','oms_gmv_6d','oms_gmv_5d','oms_gmv_4d',
#              'oms_gmv_3d','oms_gmv_2d','oms_gmv_1d','oms_gmv_today']] 

# File3 gmv ver
# = df_tmall[['oms_gmv_today','oms_gmv_1d','oms_gmv_2d','oms_gmv_3d','oms_gmv_4d','oms_gmv_5d','oms_gmv_6d'
#             ,'oms_gmv_7d','oms_gmv_8d','oms_gmv_9d','oms_gmv_10d']] 

#Create y
y = df_tmall['tmall_revenue']

# y = df_tmall.iloc[:,1] #自变量为第2列数据
# X = df_tmall.iloc[:,2:6] #自变量为第3列到第6列数据


# In[15]:


X.head()


# # Split Data

# In[16]:


# data split
X_train, X_test, y_train, y_test = train_test_split(
  X, y , random_state = 0,test_size=0.1)


# In[17]:


X_train.head()


# # Regressor model

# In[18]:


# Regressor model
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[19]:


# Prediction result
y_pred_test = regressor.predict(X_test)     # predicted value of y_test
y_pred_train = regressor.predict(X_train)   # predicted value of y_train


# In[20]:


# Regressor coefficients and intercept
# coefficients是参数，b1,b2,etc.
# intercept是截距，when x and y = 0 -> value

print(f'Coefficient: {regressor.coef_}') #斜率
print(f'Intercept: {regressor.intercept_}') #截距


# In[21]:


y_train = y_train.to_numpy(dtype = int, copy = False)
y_train.dtype

y_test = y_test.to_numpy(dtype = int, copy = False)
y_test.dtype


# In[22]:


y_pred_train = y_pred_train.astype(int)
y_pred_test = y_pred_test.astype(int)


# In[23]:


print(y_train)


# In[24]:


print(y_pred_train)


# In[25]:


print(y_test)


# In[26]:


print(y_pred_test)


# # Evaluate Diff between actual and predict

# In[27]:


# test group diff
diff_test = y_pred_test - y_test
print(diff_test)


# In[28]:


# train group diff
diff_train = y_pred_train - y_train
print(diff_train)


# ## Draw histogram graph

# In[29]:


# Plot histogram chart of train group diffs 绘画直方图，分布图
import matplotlib.pyplot as plt

plt.hist(diff_train, bins=20)
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.title('Histogram of Differences')
plt.show


# In[30]:


# Plot histogram chart of test group diffs 绘画直方图，分布图
import matplotlib.pyplot as plt

plt.hist(diff_test, bins=20)
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.title('Histogram of Differences')
plt.show


# ## Draw scatter plot

# In[31]:


# sctter plot of actual and predict values 绘制预测值与预测值的散点图
plt.scatter(y_train, y_pred_train)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot')
plt.show


# In[32]:


# plt.figure(figsize=(15,5))
# train group # 横轴是样本数，纵轴是tmall revenue
plt.plot(range(len(y_train)), y_train, 'r', label='train_data')
plt.plot(range(len(y_train)), y_pred_train, 'b', label='predict_data')
plt.legend()
plt.show()


# In[33]:


# plt.figure(figsize=(15,5))
# test group 
plt.plot(range(len(y_test)), y_test, 'r', label='test_data')
plt.plot(range(len(y_test)), y_pred_test, 'b', label='predict_data')
plt.legend()
plt.show()


# # R Square

# In[34]:


#R SQURE
from sklearn.metrics import r2_score

# rsq_train = regressor.score(X_train, y_train) 
r2_train = r2_score(y_train,y_pred_train) # r2 = r2_score(y_true, y_pred)
r2_test = r2_score(y_test,y_pred_test) 

print(r2_train)
print(r2_test)


# # Mean squared error(not sure what it means for now

# In[35]:


# Calculate mean_squared_error of test group # 计算测试组均方误差
from sklearn.metrics import mean_squared_error

MSE_test = mean_squared_error(y_test, y_pred_test)
MSE_train = mean_squared_error(y_train, y_pred_train)

RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))


# In[36]:


print('MSE train:', MSE_train)
print('RMSE train:', RMSE_train)
print('MSE test:', MSE_test)
print('RMSE test:', RMSE_test)


# # Validate Predicted Revenue

# In[37]:


df_today = pd.read_csv('./Desktop/Tmall Integration test.csv') 


# In[38]:


df_today = df_today[['receive_pack_num_9d', 'receive_pack_num_8d',
            'receive_pack_num_4d','receive_pack_num_today']]


# In[39]:


pred_today = regressor.predict(df_today)


# In[40]:


print(pred_today)


# # try R code here & print model evaluation metrics result

# In[41]:


import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

df_tmall_r = pd.read_csv('./Desktop/Tmall Integration_only_receive.csv') #比较短的发货和gmvdata


# In[42]:


# 可以直接用这个建模，也可以如下面cells分别定义x和y以后建模
# r_results = smf.ols('tmall_revenue ~ receive_pack_num_10d + receive_pack_num_9d + receive_pack_num_8d + receive_pack_num_7d + receive_pack_num_6d + receive_pack_num_5d + receive_pack_num_4d + receive_pack_num_3d + receive_pack_num_2d + receive_pack_num_1d + receive_pack_num_today', data=df_tmall_r).fit()


# In[43]:


# File2 receive package ver
X = df_tmall[['receive_pack_num_9d','receive_pack_num_8d','receive_pack_num_4d','receive_pack_num_today']] #'avg_price_per_merch_10d','oms_gmv_10d',

#Create y
y = df_tmall['tmall_revenue']


# In[44]:


#coef列表示回归系数，const表示截距 
# R²值为说明大约%的因变量变化可以由这些变量来解释
# P值表示自变量和因变量之间是否有相关性。P越小，越拒绝原假设(原假设认为自变量和因变量之间独立，不相互影响)
X2 = sm.add_constant(X)
r_results = sm.OLS(y, X2)
r_results_2 = r_results.fit()
print(r_results_2.summary())


# In[45]:


# use R model to predict values
#predict(r_results, newdata)


# In[46]:


# view usable functions of R packages
# dir(r_results)


# In[ ]:




