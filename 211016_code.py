#!/usr/bin/env python
# coding: utf-8

# ## 1- Import packages

# In[1]:


import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# ## 2- Load the data

# In[2]:


housing= pd.read_csv("/Users/AA/Desktop/Python projects/Kaggle king town housing prediction /kc_housing.csv")
housing.head()


# ## 3- Quick understanding of the data

# #### a- table structure 

# In[3]:


housing.head()


# #### b- infromation on columns 

# In[4]:


housing.info()


# #### c- aggregate information on each column 

# In[5]:


housing.describe()


# #### d- histogram of all the columns

# In[6]:


housing.hist(figsize=(20,20), grid=False, bins=50)


# ## 4- Create test set

# #### stratified sampling
# since sqft_living is highly correlated to the price, we might want to do stratified sampling based on that variable

# In[7]:


# create bins for the stratefied sampling
housing["area_category"]=pd.cut(housing.sqft_living,bins=[0,1000,2000,3000,4000, np.inf],labels=['1','2','3','4','5'])


# In[8]:


# use the Sklearn StratifiedShuffleSplit
split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)


# In[9]:


# Split the data by the area_category column 
for housing_train_set_index, housing_test_set_index in split.split(housing, housing["area_category"]):
    housing_train_set= housing.loc[housing_train_set_index]
    housing_test_set= housing.loc[housing_test_set_index]


# In[10]:


for i in housing_train_set, housing_test_set:
    i.drop("area_category", axis=1, inplace=True)


# In[11]:


# create a copy of the train set 
housing= housing_train_set.copy()


# ## 5- Discover and visualize to gain insight

# #### a- Plotting long and lat points

# In[12]:


housing.plot(kind="scatter", x="long", y="lat", alpha=0.2, figsize=(18,9), fontsize=10)


# #### b- Plotting long and lat points and price level

# In[13]:


housing.plot(kind="scatter", x="long", y="lat", alpha=1, figsize=(18,9), fontsize=10, c="price",  cmap=plt.get_cmap(), colorbar= True, s=500)


# #### c- Plot correlation matrix 

# In[14]:


corr_matrix=housing.corr()


# #### c.a-Plot as a simple table 

# In[15]:


corr_matrix["price"].sort_values(ascending=False)


# #### c.b-Plot graph

# In[16]:


plt.subplots(figsize=(20,20))
sns.heatmap(corr_matrix, vmax=1, square=True, annot= True)


# #### d- Plotting scatter plots between highly correlated attributes

# In[17]:


attributes=["price", "sqft_living", "grade", "sqft_above", "sqft_living15"]


# In[18]:


scatter_matrix(housing[attributes], figsize=(15,6))


# ## 6- Prepare the data for machine learning algorithms

# #### drop the column to predict 

# #### a- dataframe without the dependent column

# In[19]:


housing= housing_train_set.drop("price", axis=1)


# In[20]:


housing


# #### b- the dependent column

# In[21]:


housing_label= housing_train_set["price"].copy()


# In[22]:


housing_label


# ## 7- Create pipeline  

# #### a- create a column dropper class to add to the pipeline 

# In[23]:


class columnDropperTransformer():
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        return X.drop(self.columns,axis=1)

    def fit(self, X, y=None):
        return self 

transformer = FunctionTransformer(np.log1p, validate=True)


# #### b- drop unecessary columns

# In[24]:


housing_dropper=columnDropperTransformer(["date", "lat", "long", "id"])
housing_q=housing_dropper.transform(housing)


# ## 8- Feature scaling

# #### a- create a pipeline 
# - add the custom transformer class you created above and specify its arguments 
# - import StandardScaler and include it in the pipline

# In[25]:


num_pipline= Pipeline([("column_droper",columnDropperTransformer(["date", "lat", "long", "id", ])),
                       ("std_scaler", StandardScaler())])


# #### b- use the pipeline to fit the data

# In[26]:


housing= num_pipline.fit_transform(housing)


# ## 9- Apply machine learning models 

# #### a- apply the decision tree regressor ML model and show the performance

# In[27]:


tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing, housing_label)

#now lets train the model
pricing_prediction = tree_reg.predict(housing)

score=r2_score(housing_label,pricing_prediction)
print("r2 score= ", score)
print("mean_sqrd_error=",mean_squared_error(housing_label,pricing_prediction))
print("root_mean_squared error=",np.sqrt(mean_squared_error(housing_label,pricing_prediction)))
print("mean_absolute_error=",mean_absolute_error(housing_label,pricing_prediction))


# #### c- show the real prices and the predictes prices 

# In[28]:


tree_reg.fit(housing, housing_label)
#now lets train the model
pricing_predictions= tree_reg.predict(housing)
pd.DataFrame(zip(housing_label, pricing_predictions), columns=["Price_real", "Price_prediction"])

