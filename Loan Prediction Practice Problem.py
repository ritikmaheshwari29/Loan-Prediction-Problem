#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np                     # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings                        # To ignore any warnings warnings.filterwarnings("ignore")


# ## Reading and Loading the data

# In[10]:


train=pd.read_csv(r"D:\Analytics Vidhya\Loan Problem\train_ctrUa4K.csv")
test=pd.read_csv(r"D:\Analytics Vidhya\Loan Problem\test_lAUu6dG.csv")

#making a copy of train and test data
train_original=train.copy()
test_original=test.copy()


# In[14]:


# features present in data sets
train.columns    #having 12 individual datasets and 1 target dataset('Loan_Status')


# In[15]:


test.columns # lOan status is not present in dataset


# In[16]:


train.dtypes # data types for train dataset


# In[20]:


# Shape of train data set
train.shape


# In[21]:


# Shape of test data set
test.shape


# ## Univariate Analysis

# In[22]:


train['Loan_Status'].value_counts()


# In[25]:


train['Loan_Status'].value_counts(normalize=True)


# In[27]:


train['Loan_Status'].value_counts().plot.bar() #plotting bar graph


# The loan of 422(around 69%) people out of 614 was approved. 

# In[34]:


# Categorical
plt.subplot(train['Gender'].value_counts(normalize=True).plot.bar(figsize=(5,2), title= 'Gender')) 
plt.show()
plt.subplot(train['Married'].value_counts(normalize=True).plot.bar(title= 'Married'))
plt.show()
plt.subplot(train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')) 
plt.show()
plt.subplot(train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History'))
plt.show()


# It can be inferred from the above bar plots that:
# 
# 80% applicants in the dataset are male.
# Around 65% of the applicants in the dataset are married.
# Around 15% applicants in the dataset are self employed.
# Around 85% applicants have repaid their debts.

# In[40]:


# ordinal
plt.subplot(train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(5,4), title= 'Dependents'))
plt.show()
plt.subplot(train['Education'].value_counts(normalize=True).plot.bar(title= 'Education'))
plt.show()
plt.subplot(train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')) 
plt.show()


# 
# 
# Most of the applicants don’t have any dependents.
# Around 80% of the applicants are Graduate.
# Most of the applicants are from Semiurban area.

# In[41]:


# Numerical
plt.subplot(sns.distplot(train['ApplicantIncome']));
plt.show()
plt.subplot(train['ApplicantIncome'].plot.box(figsize=(16,5)))
plt.show()


# In[44]:


train.boxplot(column='ApplicantIncome', by = 'Education') 
plt.suptitle("")


#  there are a higher number of graduates with very high incomes, which are appearing to be the outliers.

# In[45]:


# Coapplicant income distribution
plt.subplot(sns.distplot(train['CoapplicantIncome']));
plt.show()
plt.subplot(train['CoapplicantIncome'].plot.box(figsize=(16,5))) 
plt.show()


# In[ ]:





# In[ ]:




