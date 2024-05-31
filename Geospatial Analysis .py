#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3


# In[48]:


import sqlite3


# In[49]:


con = sqlite3.connect(r"C:\Users\hp\OneDrive\Desktop\Sales Data\zomato_rawdata.sqlite")


# In[50]:


pd.read_sql_query("SELECT * FROM Users" , con).head(2)


# In[51]:


df = pd.read_sql_query("SELECT * FROM Users" , con)


# In[52]:


df.shape


# In[53]:


df.columns


# In[54]:


df.isnull()


# In[55]:


df.isnull().sum()


# In[56]:


df.isnull().sum()/len(df)*100


# In[57]:


df['rate'].replace(('NEW' , '-'), np.nan , inplace=True)


# In[58]:


df['rate'].unique()


# In[59]:


"4.1/5".split('/')[0]


# In[ ]:





# In[60]:


type("4.1/5".split('/')[0])


# In[61]:


float("4.1/5".split('/')[0])


# In[66]:


df['rate'] = df['rate'].apply(lambda x: float(x.split('/')[0]) if isinstance(x, str) else x)


# In[67]:


df['rate']


# In[68]:


x = pd.crosstab(df['rate'] , df['online_order'])


# In[69]:


x


# In[70]:


x.plot(kind='bar', stacked=True)


# In[71]:


x.sum(axis=1).astype(float)


# In[72]:


normalize_df = x.div(x.sum(axis=1).astype(float) , axis=0)


# In[73]:


normalize_df


# In[75]:


(normalize_df*100).plot(kind='bar', stacked=True)


# In[76]:


df['rest_type'].isnull().sum()


# In[77]:


data = df.dropna(subset=['rest_type'])


# In[78]:


data['rest_type'].isnull().sum()


# In[79]:


data['rest_type'].unique()


# In[80]:


quick_bites_df = data[data['rest_type'].str.contains('Quick Bites')]


# In[81]:


quick_bites_df.shape


# In[83]:


quick_bites_df.columns


# In[84]:


quick_bites_df['reviews_list']


# In[85]:


quick_bites_df['reviews_list'] = quick_bites_df['reviews_list'].apply(lambda x:x.lower())

from nltk.corpus import RegexpTokenizer
# In[89]:


tokenizer = RegexpTokenizer("[a-zA-Z]+")


# In[90]:


tokenizer


# In[91]:


quick_bites_df['reviews_list']


# In[92]:


tokenizer.tokenize(quick_bites_df['reviews_list'][3])


# In[93]:


sample = data [0:1000]


# In[95]:


review_tokens = sample['reviews_list'].apply(tokenizer.tokenize)


# In[96]:


review_tokens


# In[ ]:




