#!/usr/bin/env python
# coding: utf-8

# # A. Import Files

# In[1243]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression


# In[1244]:


# check the current file path
import os
os.getcwd()


# In[1245]:


# check if the file path exists
filepath = './Desktop/PFC Data.csv'
if not os.path.exists(filepath):
    print('file not exist!')
else: 
    print('file exist.')


# In[1246]:


#注意要删掉°C符号，否则运行不成功
df_pfc = pd.read_csv('./Desktop/PFC Data.csv')


# In[1247]:


# Change eventdate data type
df_pfc['Event_Date'] = pd.to_datetime(df_pfc['Event_Date'])

#Drop Typhoon impacted days
df_pfc = df_pfc.drop(df_pfc[df_pfc['Event_Date'] == '2024-09-16'].index)
# 其他Drop的方法
#X = X.drop(['NEW_COL'],axis = 1) # axis默认为0，代表删除行。如果要删除columns要指定axis = 0

#IF want to drop more columns
#df_pfc = df_pfc.drop(df_pfc[(df_pfc['Event_Date'] == '2024-09-16')|(df_pfc['Event_Date'] == '2024-09-17')].index)


# In[1248]:


#看看有没有17行；the method of selecting df by filters 
df_pfc1 = df_pfc.loc[df_pfc['Event_Date'] == '2024-09-17',:]
df_pfc1.head()


# # B. View Data

# In[1249]:


df_pfc.head()


# In[1250]:


df_pfc.tail()


# In[1251]:


df_pfc.describe()


# In[1252]:


#check data type
df_pfc.info()


# In[1253]:


# Change data type
df_pfc[['SH_Mix','NP_Mix','OC_Mix','GC_Mix','Int_Mix','NP&OC_Mix']] = df_pfc[['SH_Mix','NP_Mix','OC_Mix','GC_Mix','Int_Mix','NP&OC_Mix']].astype('float32')


# # C. Correlation Matrix

# ### 1. Create New Dataframe for Correlation Matrix

# In[1254]:


# Create a new dataframe for matrix calculation
df_pfc['Total_DPA_per_cap'] = df_pfc['On_day_DPA_per_cap'] + df_pfc['Pre_arrival_DPA_per_cap'] + df_pfc['DDP_per_cap']
df_pfc['Total_DPA_per_cap_excl_DDP'] = df_pfc['On_day_DPA_per_cap'] + df_pfc['Pre_arrival_DPA_per_cap']
df_pfc['DPA_total_Quantity_excl_DDP'] = df_pfc['DPA_total_excl._DDP']

df_pfc['SH&GC&Int_Mix'] = df_pfc['SH_Mix'] + df_pfc['GC_Mix'] + df_pfc['Int_Mix']

df_pfc_matrix_full = df_pfc[['Total_DPA_per_cap','EPEP_per_cap','On_day_DPA_per_cap','Pre_arrival_DPA_per_cap','DDP_per_cap',
                       'DPA_total_Quantity','EPEP_Quantity','On_day_DPA_Quantity','Pre_arrival_DPA_Quantity','DDP_Quantity',
                        'Marquee','Paid_Attendance','Total_Attendance','Paid_Mix',
                       'SH_Mix','NP_Mix','OC_Mix','GC_Mix','Int_Mix',
                        'NP&OC_Mix', 'SH&GC&Int_Mix',
                        'Paid_RPA(excl._Select)']]
# Per cap matrix

df_pfc_matrix_DPA_per_cap = df_pfc[['Total_DPA_per_cap',
                        'Marquee','Paid_Attendance','Total_Attendance','Paid_Mix', 'NP&OC_Mix', 'OC_Mix','Paid_RPA(excl._Select)']]

df_pfc_matrix_EPEP_per_cap = df_pfc[['EPEP_per_cap',
                        'Marquee','Paid_Attendance','Total_Attendance','Paid_Mix', 'NP&OC_Mix', 'OC_Mix','Paid_RPA(excl._Select)']]

df_pfc_matrix_onday_DPA_per_cap = df_pfc[['On_day_DPA_per_cap',
                        'Marquee','Paid_Attendance','Total_Attendance','Paid_Mix', 'NP&OC_Mix', 'OC_Mix','Paid_RPA(excl._Select)']]

df_pfc_matrix_prearrival_DPA_per_cap = df_pfc[['Pre_arrival_DPA_per_cap',
                        'Marquee','Paid_Attendance','Total_Attendance','Paid_Mix', 'NP&OC_Mix', 'OC_Mix','Paid_RPA(excl._Select)']]

df_pfc_matrix_DDP_per_cap = df_pfc[['DDP_per_cap',
                        'Marquee','Paid_Attendance','Total_Attendance','Paid_Mix', 'NP&OC_Mix', 'OC_Mix','Paid_RPA(excl._Select)']]


# Qty matrix

df_pfc_matrix_DPA_qty = df_pfc[['DPA_total_Quantity',
                        'Marquee','Paid_Attendance','Total_Attendance', 'NP&OC_Mix']]

df_pfc_matrix_EPEP_qty = df_pfc[['EPEP_Quantity',
                        'Marquee','Paid_Attendance','Total_Attendance', 'NP&OC_Mix']]

df_pfc_matrix_onday_DPA_qty = df_pfc[['On_day_DPA_Quantity',
                        'Marquee','Paid_Attendance','Total_Attendance', 'NP&OC_Mix']]

df_pfc_matrix_prearrival_DPA_qty = df_pfc[['Pre_arrival_DPA_Quantity',
                        'Marquee','Paid_Attendance','Total_Attendance', 'NP&OC_Mix']]

df_pfc_matrix_DDP_qty = df_pfc[['DDP_Quantity',
                        'Marquee','Paid_Attendance','Total_Attendance', 'NP&OC_Mix']]

# delete 'Paid_Mix','DoW_num',


# In[1255]:


df_pfc_matrix_full.head()


# ### 2. Correlation Matrix for All Key Influencers

# In[1256]:


# Correlation Matrix
corr_full = df_pfc_matrix_full.corr()
corr_full.style.background_gradient(cmap='RdBu')
# Conclusion: DoW & no correlation with PFC spending per cap; total attendance and paid both have strong relationship with 
# per cap, can keep one only


# ### 3. Correlation Matrix with fewer metrics for deck

# In[1257]:


# Create a smaller matrix for deck - DPA per cap
corr_DPA_per_cap = df_pfc_matrix_DPA_per_cap.corr()
corr_DPA_per_cap.style.background_gradient(cmap='Wistia')


# In[1258]:


# Create a smaller matrix for deck - EPEP per cap
corr_EPEP_per_cap = df_pfc_matrix_EPEP_per_cap.corr()
corr_EPEP_per_cap.style.background_gradient(cmap='Wistia')


# In[1259]:


# Create a smaller matrix for deck - Onday DPA per cap
corr_EPEP_per_cap = df_pfc_matrix_onday_DPA_per_cap.corr()
corr_EPEP_per_cap.style.background_gradient(cmap='Wistia')


# In[1260]:


# Create a smaller matrix for deck - Prearrival DPA per cap
corr_EPEP_per_cap = df_pfc_matrix_prearrival_DPA_per_cap.corr()
corr_EPEP_per_cap.style.background_gradient(cmap='Wistia')


# In[1261]:


# Create a smaller matrix for deck - DDP per cap
corr_EPEP_per_cap = df_pfc_matrix_DDP_per_cap.corr()
corr_EPEP_per_cap.style.background_gradient(cmap='Wistia')


# In[ ]:





# In[1262]:


# Create a smaller matrix for deck - DPA qty
corr_DPA_qty = df_pfc_matrix_DDP_per_cap.corr()
corr_DPA_qty.style.background_gradient(cmap='coolwarm')


# In[1263]:


# Create a smaller matrix for deck - EPEP qty
corr_EPEP_qty = df_pfc_matrix_EPEP_qty.corr()
corr_EPEP_qty.style.background_gradient(cmap='coolwarm')


# In[1264]:


# Create a smaller matrix for deck - Onday DPA qty
corr_EPEP_qty = df_pfc_matrix_onday_DPA_qty.corr()
corr_EPEP_qty.style.background_gradient(cmap='coolwarm')


# In[1265]:


# Create a smaller matrix for deck - Prearrival DPA qty
corr_EPEP_qty = df_pfc_matrix_prearrival_DPA_qty.corr()
corr_EPEP_qty.style.background_gradient(cmap='coolwarm')


# In[1266]:


# Create a smaller matrix for deck - DDP qty
corr_EPEP_qty = df_pfc_matrix_DDP_qty.corr()
corr_EPEP_qty.style.background_gradient(cmap='coolwarm')


# ### 4. The correlation of Paid RPA and per cap in same marquee

# In[1267]:


df_pfc_matrix_475 = df_pfc.loc[df_pfc['Marquee'] == 475,:]
df_pfc_matrix_475 = df_pfc_matrix_475[['Total_DPA_per_cap','EPEP_per_cap','On_day_DPA_per_cap','Pre_arrival_DPA_per_cap','DDP_per_cap',
                       'DPA_total_Quantity','EPEP_Quantity',
                        'Paid_RPA(excl._Select)']]

df_pfc_matrix_599 = df_pfc.loc[df_pfc['Marquee'] == 599,:]
df_pfc_matrix_599 = df_pfc_matrix_599[['Total_DPA_per_cap','EPEP_per_cap','On_day_DPA_per_cap','Pre_arrival_DPA_per_cap','DDP_per_cap',
                       'DPA_total_Quantity','EPEP_Quantity',
                        'Paid_RPA(excl._Select)']]

df_pfc_matrix_719 = df_pfc.loc[df_pfc['Marquee'] == 719,:]
df_pfc_matrix_719 = df_pfc_matrix_719[['Total_DPA_per_cap','EPEP_per_cap','On_day_DPA_per_cap','Pre_arrival_DPA_per_cap','DDP_per_cap',
                       'DPA_total_Quantity','EPEP_Quantity',
                        'Paid_RPA(excl._Select)']]

df_pfc_matrix_799 = df_pfc.loc[df_pfc['Marquee'] == 799,:]
df_pfc_matrix_799 = df_pfc_matrix_799[['Total_DPA_per_cap','EPEP_per_cap','On_day_DPA_per_cap','Pre_arrival_DPA_per_cap','DDP_per_cap',
                       'DPA_total_Quantity','EPEP_Quantity',
                        'Paid_RPA(excl._Select)']]


# In[1268]:


corr_475 = df_pfc_matrix_475.corr()
corr_475.style.background_gradient(cmap='coolwarm')


# In[1269]:


corr_599 = df_pfc_matrix_599.corr()
corr_599.style.background_gradient(cmap='coolwarm')


# In[1270]:


corr_719 = df_pfc_matrix_719.corr()
corr_719.style.background_gradient(cmap='coolwarm')


# In[1271]:


corr_799 = df_pfc_matrix_799.corr()
corr_799.style.background_gradient(cmap='coolwarm')


# #### Result
# Paid RPA does not have strong relationship with PFC spending, at least from current data cannot be proven. 
# Maybe can explore "Promo%" and "Discount%"'s relationship with spending per cap next

# # D. View Correlation through Scatter Plot

# ## 1. DPA per Cap

# In[1272]:


import matplotlib.pyplot as plt #Matplotlib – Python画图工具库
import seaborn as sns #Seaborn – 统计学数据可视化工具库
# show scatter plot bettween multiple independent variables and the dependent variable
sns.pairplot(df_pfc_matrix_DPA_per_cap, x_vars=['Marquee','Paid_Attendance','Total_Attendance', 'NP&OC_Mix'], 
                          y_vars='Total_DPA_per_cap', 
                          height=4, aspect=1, kind='scatter')
plt.show() # show plot


# ### 1.1 看看DPA per Cap和平方之间是不是非线性关系

# In[1273]:


## Tried the power of 1.5, 2, 3, 4 on total and paid attendance. Among which, the 3rd power has the besting fitting result.
## 分别对总人数和付费人数尝试了1.5, 2, 3, 4的幂次方，其中3次方的拟合效果最好。
df_pfc_matrix_DPA_per_cap.loc[:,'Total_Attendance_3'] = df_pfc_matrix_DPA_per_cap['Total_Attendance']**3
df_pfc_matrix_DPA_per_cap.loc[:,'Paid_Attendance_3'] = df_pfc_matrix_DPA_per_cap['Paid_Attendance']**3
df_pfc_matrix_DPA_per_cap.head()


# In[1274]:


import matplotlib.pyplot as plt #Matplotlib – Python画图工具库
import seaborn as sns #Seaborn – 统计学数据可视化工具库
# show scatter plot bettween multiple independent variables and the dependent variable
sns.pairplot(df_pfc_matrix_DPA_per_cap, x_vars=['Marquee','Paid_Attendance','Paid_Attendance_3',
                                                'Total_Attendance','Total_Attendance_3',
                                                'NP&OC_Mix'], 
                          y_vars='Total_DPA_per_cap', 
                          height=4, aspect=1, kind='scatter')
plt.show() # show plot


# In[1275]:


# For deck's charts
sns.pairplot(df_pfc_matrix_DPA_per_cap, x_vars=['Marquee','Paid_Attendance_3','Total_Attendance_3', 'NP&OC_Mix'], 
                          y_vars='Total_DPA_per_cap', 
                          height=4, aspect=1, kind='scatter')
plt.show() # show plot


# ## 2. EPEP per cap

# In[1276]:


sns.pairplot(df_pfc_matrix_EPEP_per_cap, x_vars=['Marquee','Paid_Attendance','Total_Attendance', 'NP&OC_Mix'], 
                          y_vars='EPEP_per_cap', 
                          height=4, aspect=1, kind='scatter')
plt.show() # show plot


# ## 3. DPA Qty

# In[1277]:


sns.pairplot(df_pfc_matrix_DPA_qty, x_vars=['Marquee','Paid_Attendance','Total_Attendance', 'NP&OC_Mix'], 
                          y_vars='DPA_total_Quantity', 
                          height=4, aspect=1, kind='scatter')
plt.show() # show plot


# ## 4. EPEP Qty

# In[1278]:


sns.pairplot(df_pfc_matrix_EPEP_qty, x_vars=['Marquee','Paid_Attendance','Total_Attendance', 'NP&OC_Mix'], 
                          y_vars='EPEP_Quantity', 
                          height=4, aspect=1, kind='scatter')
plt.show() # show plot


# # E. Linear Regression

# # E1. DPA per Cap

# ## 1. Python Model

# ### 1.1 Align X and y

# In[1279]:


# Align X and y
#Create X
X = df_pfc[['Marquee','Paid_Attendance','Total_Attendance', 'NP&OC_Mix']]

X.loc[:,'Paid_Attendance_3'] = X['Paid_Attendance']**3

X.loc[:,'Total_Attendance_3'] = X['Total_Attendance']**3

X = X.drop(['Paid_Attendance'],axis = 1)

X = X.drop(['Total_Attendance'],axis = 1)

X = X.drop(['NP&OC_Mix'],axis = 1) # 发现去掉Mix以后R方也没有明显下降，Mix对于DPA per cap的影响不大

#Create y
y = df_pfc['Total_DPA_per_cap']

# y = df_tmall.iloc[:,1] #自变量为第2列数据
# X = df_tmall.iloc[:,2:6] #自变量为第3列到第6列数据


# In[1280]:


X.head()


# ### 1.2 Split Data

# In[1281]:


# data split
X_train, X_test, y_train, y_test = train_test_split(
  X, y , random_state = 0,test_size=0.1)


# In[1282]:


X_train.head()


# ### 1.3 Regressor model

# In[1283]:


# Regressor model
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[1284]:


# Prediction result
y_pred_test = regressor.predict(X_test)     # predicted value of y_test
y_pred_train = regressor.predict(X_train)   # predicted value of y_train


# In[1285]:


# Regressor coefficients and intercept
# coefficients是参数，b1,b2,etc.
# intercept是截距，when x and y = 0 -> value

print(f'Coefficient: {regressor.coef_}') #斜率
print(f'Intercept: {regressor.intercept_}') #截距


# ## 2. try R code here & print model evaluation metrics result

# In[1286]:


import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

df_pfc_r = pd.read_csv('./Desktop/PFC Data.csv')


# In[1287]:


# 可以直接用这个建模，也可以如下面cells分别定义x和y以后建模
# r_results = smf.ols('tmall_revenue ~ receive_pack_num_10d + receive_pack_num_9d + receive_pack_num_8d + receive_pack_num_7d + receive_pack_num_6d + receive_pack_num_5d + receive_pack_num_4d + receive_pack_num_3d + receive_pack_num_2d + receive_pack_num_1d + receive_pack_num_today', data=df_tmall_r).fit()


# In[1288]:


# File2 receive package ver
XR = df_pfc[['Marquee','Paid_Attendance','Total_Attendance', 'NP&OC_Mix']]


#Create y
yR = df_pfc['Total_DPA_per_cap']


# In[1289]:


XR.loc[:,'Paid_Attendance_3'] = XR['Paid_Attendance']**3
XR.loc[:,'Total_Attendance_3'] = XR['Total_Attendance']**3

XR = XR.drop(['Paid_Attendance'],axis = 1)
XR = XR.drop(['Total_Attendance'],axis = 1)
XR = XR.drop(['NP&OC_Mix'],axis = 1)

XR.head()


# In[1290]:


#coef列表示回归系数，const表示截距 
# R²值为说明大约%的因变量变化可以由这些变量来解释
# P值表示自变量和因变量之间是否有相关性。P越小，越拒绝原假设(原假设认为自变量和因变量之间独立，不相互影响)
X2 = sm.add_constant(XR)
r_results = sm.OLS(yR, X2)
r_results_2 = r_results.fit()
print(r_results_2.summary())


# In[1291]:


# use R model to predict values
#predict(r_results, newdata)


# In[1292]:


# view usable functions of R packages
# dir(r_results)


# ## 3. Evaluate Diff between actual and predict - for Python Linear Regression Model

# ## 3.1 Cal Diff

# In[1293]:


# Use the forest's predict method on the train data
# Calculate the absolute errors
errors_train_linear = (y_pred_train - y_train)
# Print out the mean absolute error (mae)
print('Mean Absolute Error of train group:', round(np.mean(errors_train_linear), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape_train_linear = 100 * abs(errors_train_linear / y_train)
# Calculate and display accuracy
accuracy_train_linear = 100 - np.mean(mape_train_linear)
print('Accuracy of train group:', round(accuracy_train_linear, 2), '%.')


# In[1294]:


# Use the forest's predict method on the test data
# Calculate the absolute errors
errors_test_linear = (y_pred_test - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error of test group:', round(np.mean(errors_test_linear), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape_test_linear = 100 * abs(errors_test_linear / y_test)
# Calculate and display accuracy
accuracy_test_linear = 100 - np.mean(mape_test_linear)
print('Accuracy of test group:', round(accuracy_test_linear, 2), '%.')


# In[1295]:


y_train = y_train.to_numpy(dtype = float, copy = False)
y_train.dtype

y_test = y_test.to_numpy(dtype = float, copy = False)
y_test.dtype


# In[1296]:


y_pred_train = y_pred_train.astype(float)
y_pred_test = y_pred_test.astype(float)


# In[1297]:


print(y_train)


# In[1298]:


print(y_pred_train)


# In[1333]:


print(X_test)


# In[1340]:


X = df_pfc[['Marquee','Paid_Attendance','Total_Attendance', 'NP&OC_Mix']]
row_data = X.loc[106]
print(row_data)


# In[1299]:


print(y_test)


# In[1300]:


print(y_pred_test)


# In[1301]:


# test group diff
diff_test = y_pred_test - y_test
print(diff_test)


# In[1302]:


# train group diff
diff_train = y_pred_train - y_train
print(diff_train)


# ## 3.2 Draw histogram graph

# In[1303]:


# Plot histogram chart of train group diffs 绘画直方图，分布图
import matplotlib.pyplot as plt

plt.hist(diff_train, bins=20)
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.title('Histogram of Differences')
plt.show


# In[1304]:


# Plot histogram chart of test group diffs 绘画直方图，分布图
import matplotlib.pyplot as plt

plt.hist(diff_test, bins=20)
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.title('Histogram of Differences')
plt.show


# ## 3.3 Draw scatter plot

# ### 3.3.1 A. Result when considering non-linear relationship

# In[1305]:


# sctter plot of actual and predict values 绘制真实值与预测值的散点图
plt.scatter(y_train, y_pred_train)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot')
plt.show


# In[1306]:


# sctter plot of actual and predict values 绘制真实值与预测值的散点图
plt.scatter(y_test, y_pred_test)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot')
plt.show


# ### 3.3.1 B. Result when not considering non-linear relationship <- for comparison

# In[1307]:


# Align X and y
#Create X
X_compare = df_pfc[['Marquee','Paid_Attendance','Total_Attendance', 'NP&OC_Mix']]

#Create y
y_compare = df_pfc['Total_DPA_per_cap']

#Data split
X_train_compare, X_test_compare, y_train_compare, y_test_compare = train_test_split(
  X_compare, y_compare , random_state = 0,test_size=0.1)

# Regressor model
regressor_compare = LinearRegression()
regressor.fit(X_train_compare, y_train_compare)

# Prediction result
y_pred_test_compare = regressor.predict(X_test_compare)     # predicted value of y_test
y_pred_train_compare = regressor.predict(X_train_compare)   # predicted value of y_train

# Transfer dtype
y_train_compare = y_train_compare.to_numpy(dtype = float, copy = False)
y_test_compare = y_test_compare.to_numpy(dtype = float, copy = False)

y_pred_train_compare = y_pred_train_compare.astype(float)
y_pred_test_compare = y_pred_test_compare.astype(float)

# Cal result diff
diff_test_compare = y_pred_test_compare - y_test_compare
diff_train_compare = y_pred_train_compare - y_train_compare

# sctter plot of actual and predict values 绘制真实值与预测值的散点图
plt.scatter(y_train_compare, y_pred_train_compare)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot')
plt.show


# In[1308]:


# Histogram of Difference
plt.hist(diff_train_compare, bins=20)
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.title('Histogram of Differences')
plt.show


# In[1309]:


plt.hist(diff_test_compare, bins=20)
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.title('Histogram of Differences')
plt.show


# ### 3.3.2 Train Group accuracy (considering non-linear relationship)

# In[1310]:


# plt.figure(figsize=(15,5))
# train group # 横轴是样本数，纵轴是tmall revenue
plt.plot(range(len(y_train)), y_train, 'r', label='train_data')
plt.plot(range(len(y_train)), y_pred_train, 'b', label='predict_data')
plt.legend()
plt.show()


# ### 3.3.3 Test Group accuracy (considering non-linear relationship)

# In[1311]:


# plt.figure(figsize=(15,5))
# test group 
plt.plot(range(len(y_test)), y_test, 'r', label='test_data')
plt.plot(range(len(y_test)), y_pred_test, 'b', label='predict_data')
plt.legend()
plt.show()


# ## 3.4 R Square

# In[1312]:


#R SQURE
from sklearn.metrics import r2_score

# rsq_train = regressor.score(X_train, y_train) 
r2_train = r2_score(y_train,y_pred_train) # r2 = r2_score(y_true, y_pred)
r2_test = r2_score(y_test,y_pred_test) 

print(r2_train)
print(r2_test)


# # Validate Predicted Revenue

# In[1313]:


# df_today = pd.read_csv('./Desktop/Tmall Integration test.csv') 


# In[1314]:


# df_today = df_today[['receive_pack_num_9d', 'receive_pack_num_8d', 'receive_pack_num_4d','receive_pack_num_today']]


# In[1315]:


# pred_today = regressor.predict(df_today)


# In[1316]:


# print(pred_today)


# # F. Random Forest

# ## 1. Data Manipulation

# In[1317]:


# Create a random forest dataframe
df_pfc_rf = df_pfc[['Total_DPA_per_cap','EPEP_per_cap',
                    'DPA_total_Quantity','EPEP_Quantity',
                    'Marquee','Paid_Attendance','Total_Attendance',
                    'SH_Mix','NP_Mix','OC_Mix','GC_Mix','Int_Mix',
                    'NP&OC_Mix', 'SH&GC&Int_Mix',
                    'Paid_RPA(excl._Select)']]


df_pfc_rf.loc[:,'Total_Attendance_3'] = df_pfc_rf['Total_Attendance']**3
df_pfc_rf.loc[:,'Paid_Attendance_3'] = df_pfc_rf['Paid_Attendance']**3


df_pfc_rf_dpa_per_cap = df_pfc_rf[['Total_DPA_per_cap',
                    'Marquee','Paid_Attendance_3','Total_Attendance_3']]
df_pfc_rf_epep_per_cap = df_pfc_rf[['EPEP_per_cap',
                    'Marquee','Paid_Attendance_3','Total_Attendance_3']]
df_pfc_rf_epep_qty = df_pfc_rf[['EPEP_Quantity',
                    'Marquee','Paid_Attendance_3','Total_Attendance_3']]


# Convert Dataframe to Arrays
import numpy as np

# Labels are the values we want to predict
y = np.array(df_pfc_rf_dpa_per_cap['Total_DPA_per_cap'])
y_epep_cap = np.array(df_pfc_rf_epep_per_cap['EPEP_per_cap'])
y_epep_qty = np.array(df_pfc_rf_epep_qty['EPEP_Quantity'])

# Remove the labels from the dataframe
# axis 1 refers to the columns
X = df_pfc_rf_dpa_per_cap.drop('Total_DPA_per_cap', axis = 1)
X_epep_cap = df_pfc_rf_epep_per_cap.drop('EPEP_per_cap', axis = 1)
X_epep_qty = df_pfc_rf_epep_qty.drop('EPEP_Quantity', axis = 1)

# Saving feature names for later use
feature_list = list(X.columns)

# Convert to numpy array
X = np.array(X)
X_epep_cap = np.array(X_epep_cap)
X_epep_qty = np.array(X_epep_qty)


# In[1318]:


df_pfc_rf_dpa_per_cap.head()


# In[1319]:


# Split data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.1, random_state = 1)

print('Training X Shape:', train_X.shape)
print('Training y Shape:', train_y.shape)
print('Testing X Shape:', test_X.shape)
print('Testing y Shape:', test_y.shape)


# In[1320]:


print(test_y)


# In[1321]:


# baseline predictions (直接用test Y avg.作为预测结果，看平均Diff的大小) <-也即如果我们的模型不能超过这个表现的话 那模型不可用

# The baseline predictions are the historical averages
# 这串代码有问题，会直接替换原本的test_y

# 代码如下：
# baseline_preds = test_y
# baseline_preds[:] = train_y.mean()

# Baseline errors, and display average baseline error
# baseline_errors = abs(baseline_preds - test_y)
# print('Average baseline error: ', round(np.mean(baseline_errors), 2))


# In[1322]:


# Train Model
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 100, max_depth = 5, min_samples_leaf = 2, min_samples_split = 4,
                           # max_features = 'log2',
                           bootstrap = False, max_leaf_nodes = 20, random_state = 0)
# Train the model on training data
rf.fit(train_X, train_y)


# In[1323]:


# Use the forest's predict method on the test data
predictions_train = rf.predict(train_X)
# Calculate the absolute errors
errors_train = predictions_train - train_y
# Print out the mean absolute error (mae)
print('Mean Absolute Error of train group:', round(np.mean(errors_train), 2), 'degrees.')


# In[1324]:


# Use the forest's predict method on the test data
predictions_test = rf.predict(test_X)
# Calculate the absolute errors
errors_test = predictions_test - test_y
# Print out the mean absolute error (mae)
print('Mean Absolute Error of test group:', round(np.mean(errors_test), 2), 'degrees.')


# In[1325]:


print(predictions_test)


# In[1326]:


print(test_y)


# In[1327]:


print(errors_test)


# In[1328]:


# Calculate mean absolute percentage error (MAPE)
mape_train = 100 * abs(errors_train / train_y)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape_train)
print('Accuracy of train group:', round(accuracy, 2), '%.')


# In[1329]:


# Calculate mean absolute percentage error (MAPE)
mape_test = 100 * abs(errors_test / test_y)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape_test)
print('Accuracy of test group:', round(accuracy, 2), '%.')

# 模型效果不太好..好像是过拟合吗，怎么做


# In[1330]:


# 计算R平方值
r2 = r2_score(train_y, predictions_train)
print(f'R² Score: {r2}')


# In[1331]:


# 计算R平方值
r2 = r2_score(test_y, predictions_test)
print(f'R² Score: {r2}')


# In[1067]:


# sctter plot of actual and predict values 绘制真实值与预测值的散点图
plt.scatter(train_y, predictions_train)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot')
plt.show


# In[1068]:


# sctter plot of actual and predict values 绘制真实值与预测值的散点图
plt.scatter(test_y, predictions_test)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot')
plt.show


# In[793]:


# Plot histogram chart of train group diffs 绘画直方图，分布图
import matplotlib.pyplot as plt

plt.hist(errors_train, bins=20)
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.title('Histogram of Differences')
plt.show


# In[794]:


# Plot histogram chart of test group diffs 绘画直方图，分布图
import matplotlib.pyplot as plt

plt.hist(errors_test, bins=20)
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.title('Histogram of Differences')
plt.show


# In[795]:


# plt.figure(figsize=(15,5))
# train group # 横轴是样本数，纵轴是tmall revenue
plt.plot(range(len(train_y)), train_y, 'r', label='train_data')
plt.plot(range(len(train_y)), predictions_train, 'b', label='predict_data')
plt.legend()
plt.show()


# In[796]:


# plt.figure(figsize=(15,5))
# train group # 横轴是样本数，纵轴是tmall revenue
plt.plot(range(len(test_y)), test_y, 'r', label='train_data')
plt.plot(range(len(test_y)), predictions_test, 'b', label='predict_data')
plt.legend()
plt.show()


# In[ ]:




