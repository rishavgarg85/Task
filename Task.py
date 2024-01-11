#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[45]:


boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url)


# In[46]:


boston_df.head()


# In[47]:


ax = sns.boxplot(y ="MEDV", data=boston_df)


# In[48]:


pyplot.hist(boston_df['CHAS'])


# In[49]:


# Discretize the 'AGE' variable into three groups
boston_df['Age_Group'] = pd.cut(boston_df['AGE'], bins=[0, 35, 70, float('inf')], labels=['35 years and younger', 'Between 35 and 70 years', '70 years and older'])


# In[50]:


# Create the boxplot using Seaborn
sns.boxplot(
    x = "Age_Group",
    y = "MEDV",
    showmeans=True,  # Display mean markers within the boxes
    data=boston_df
)


# In[52]:


#Create Scatter plot
plt.scatter(boston_df['INDUS'], boston_df['NOX'], color='blue', alpha=0.7)
plt.title('Scatter Plot of NOX vs INDUS')
plt.xlabel('Proportion of Non-retail Business Acres per Town (INDUS)')
plt.ylabel('Nitric Oxide Concentrations (NOX)')
plt.grid(True)
plt.show()


# In[ ]:


# correlation coefficient of 0.76 indicates a strong positive correlation between the proportion of non-retail business acres per town (INDUS) and nitric oxide concentrations (NOX). In other words, as the proportion of non-retail business acres per town increases, the nitric oxide concentrations tend to increase as well.


# In[53]:


# Calculate and print the correlation coefficient
correlation_coefficient = boston_df['INDUS'].corr(boston_df['NOX'])
print(f'Correlation Coefficient: {correlation_coefficient:.2f}')

plt.show()


# In[58]:


# Create histogram
plt.hist(boston_df['PTRATIO'], bins=20, color='green', alpha=0.7)
plt.title('Histogram of PTRATIO in Boston')
plt.xlabel('PTRATIO')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[59]:


pip install scipy statsmodels numpy pandas matplotlib


# In[ ]:




