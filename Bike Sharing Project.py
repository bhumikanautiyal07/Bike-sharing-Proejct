#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


bikesharing_dataframe = pd.read_csv (r"C:\Users\nautiyal\Downloads\day.csv")
bikesharing_dataframe


# In[4]:


bikesharing_dataframe.head()


# In[5]:


#we can remove some unnecessary colomns
bikesharing_dataframe.drop(["instant"], axis = 1, inplace = True)
bikesharing_dataframe.head()


# In[6]:


#similarly we can drop more colomns like registered and casual as cnt already contains total users

bikesharing_dataframe.drop(["casual"], axis = 1, inplace=True)
bikesharing_dataframe.drop(["registered"], axis = 1, inplace = True)
#we can also drop teh date coloumn as we already have month and day
bikesharing_dataframe. drop(["dteday"], axis = 1, inplace = True)
bikesharing_dataframe.head()


# In[7]:


bikesharing_dataframe.corr()


# In[8]:


#number of unique entries in dataset
bikesharing_dataframe.nunique()


# In[9]:


coloumns = ["temp", "atemp", "hum", "windspeed", "cnt"]
plt.figure(figsize=(40,20))

i=1
for coloumn in coloumns:
    plt.subplot(1,5,i)
    sns.boxplot(y= coloumn, data =bikesharing_dataframe)
    i+=1

    #as we can see there are no outliers


# In[10]:


# # We can now replace season , month and day og the week from numerical to strings
bikesharing_dataframe.season.replace({1:"spring",2:"summer", 3:"fall", 4:"winter"}, inplace = True)
bikesharing_dataframe.mnth.replace({1:"jan", 2:"feb", 3:"Mar", 4:"april", 5:"may", 6:"june", 7:"july", 8:"aug", 9:"sept", 10:"oct", 11:"nov", 12:"dec"}, inplace= True)
bikesharing_dataframe.weekday.replace({1:"Monday", 2:"Tueday", 3:"Wednesday", 4:"thursday", 5:"friday", 6:"saturday", 7:"sunday"}, inplace= True)
bikesharing_dataframe.weathersit.replace({1:"clear", 2:"cloudy", 3:"bad", 4:"Rainy"}, inplace = True)

bikesharing_dataframe.head()


# In[11]:


bikesharing_dataframe.describe()


# In[12]:


bikesharing_dataframe.info()


# In[13]:


#checking null values is a necessary step as this can lead wrong predictions of the dataset
bikesharing_dataframe.isnull().sum()
#there are no null values in dataset


# In[14]:


#checking the linear relationship of variables
plt.figure(figsize = (10,20))
sns.pairplot(data= bikesharing_dataframe, vars = ["cnt", "temp", "atemp", "windspeed", "hum"])


# In[15]:


#checking linear relationship between different variables
#visualizing data to find corelation from numerical variables
plt.figure(figsize=(30,20))
sns.pairplot(data = bikesharing_dataframe)


# In[16]:


#heatmap to show corelation between numerical variables
plt.figure
sns.heatmap(bikesharing_dataframe[["cnt", "temp", "atemp", "windspeed", "hum"]].corr(), cmap = "YlGnBu",annot=True)
plt.show()


# In[17]:


numeric_df = bikesharing_dataframe.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()
sns.heatmap(corr, cmap="YlGnBu", annot = True)
plt.show()


# In[18]:


# Now for the variables with categories
plt.figure(figsize=(20,15))

plt.subplot(2,3,1)
sns.boxplot(x= "season", y  = "cnt", data =bikesharing_dataframe)

plt.subplot(2,3,2)
sns.boxplot(x="yr", y = "cnt", data= bikesharing_dataframe)

plt.subplot(2,3,3)
sns.boxplot(x= "mnth", y= "cnt", data = bikesharing_dataframe)

plt.subplot(2,3,4)
sns.boxplot(x="weekday", y="cnt", data = bikesharing_dataframe)

plt.subplot(2,3,5)
sns.boxplot(x="workingday", y= "cnt", data= bikesharing_dataframe)

plt.subplot(2,3,6)
sns.boxplot(x="weathersit", y= "cnt", data= bikesharing_dataframe)
plt.show()


# In[19]:


bikesharing_dataframe.describe()


# # Data Preparation for Linear regression

# In[20]:


dummy1 = pd.get_dummies(bikesharing_dataframe[['season', 'mnth', 'weekday']], drop_first=True)
dummy2 = pd.get_dummies(bikesharing_dataframe[["weathersit"]])
bikesharing_dataframe = pd.concat([bikesharing_dataframe, dummy1], axis=1)
bikesharing_dataframe = pd.concat([bikesharing_dataframe, dummy2], axis=1)


# In[21]:


bikesharing_dataframe.columns


# In[22]:


bikesharing_dataframe = bikesharing_dataframe.drop(['season', 'mnth', 'weekday', 'weathersit'], axis=1)


# In[23]:


bikesharing_dataframe.head()


# # Train - test split

# In[24]:


import sklearn
from sklearn.model_selection import train_test_split


# In[25]:


#y to containg target variable

y = bikesharing_dataframe.pop("cnt")


# In[26]:


x = bikesharing_dataframe
bikesharing_dataframe.head()


# In[27]:


#train-test split in 70-30 ratio
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.3, random_state = 42)


# In[28]:


x.columns


# In[29]:


x.head()


# In[30]:


#checking the shape and size for train and test
print(x_train.shape)
print(x_test.shape)


# In[31]:


#importing required library
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


# In[32]:


#continous variables
cont_var = ["temp", "atemp", "hum", "windspeed"]

scaler = MinMaxScaler()

x_train[cont_var] = scaler.fit_transform(x_train[cont_var])


# In[33]:


x_train.describe()


# In[34]:


x_train.head()


# In[35]:


plt.figure(figsize =(40,40))
sns.heatmap(x_train.corr(), annot = True)
plt.show()


# # Building model using RFE

# In[36]:


lr = LinearRegression()
lr.fit(x_train, y_train)


# In[38]:


from sklearn.feature_selection import RFE


# In[ ]:


#cut down number of feaatures t0 15 using automated approach
rfe= RFE (estimator= lr, n_features_to_select=15)
rfe.fit(x_train, y_train)


# In[46]:


#coloumns slected by RFE
list(zip(x_train.columns, rfe.support_,rfe.ranking_))


# In[59]:


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[76]:


#function to build a model
def build_model(cols):
    x_train_sm = sm.add_constant(x_train[cols])
    lm = sm.OLS(y_train, x_train_sm).fit()  
    print(lm.summary())
    return lm


# In[77]:


#to calculate VIF 
def get_vif(cols):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    x_train_sm = sm.add_constant(x_train[cols])
    vif = pd.DataFrame()
    vif['Features'] = x_train_sm.columns
    vif['VIF'] = [variance_inflation_factor(x_train_sm.values, i) for i in range(x_train_sm.shape[1])]
    return vif


# In[78]:


#printing the columns selected by RFE.
x_train.columns[rfe.support_]


# In[79]:


#not selected in RFE
x_train.columns[~rfe.support_]


# In[80]:


#but we will take 15 columns for regression
x_train_rfe = x_train[['yr', 'holiday', 'temp', 'atemp', 'hum', 'windspeed', 'season_spring','season_summer',
       'season_winter', 'mnth_jan', 'mnth_july', 'mnth_sept',
       'weekday_saturday', 'weathersit_bad', 'weathersit_cloudy']]


# In[81]:


x_train_rfe.shape


# # Model 1

# In[86]:


#columns with selected by RFE
cols = ['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jan', 'mnth_july', 'mnth_sept', 'weekday_saturday',
       'weathersit_bad', 'weathersit_cloudy']

build_model(cols)
get_vif(cols)


# In[ ]:





# In[ ]:




