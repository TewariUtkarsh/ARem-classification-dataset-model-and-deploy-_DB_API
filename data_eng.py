# Importing Libraries
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import statsmodels.api as sm
import seaborn as sns


# Loading Dataset
df = pd.read_csv('data.csv')


# Generating Profile Report
pf = ProfileReport(df)


# Saving the Profile Report
pf.to_file('report.html')


# Conclusions from EDA and Profile Report
"""
Conclusions:
1. No Missing/NaN values -> already imputed
2. time is irrelevant for prediction as per documentation -> drop time col
3. No Zero values to handle -> no imputation
4. No certain Multicollinearity -> VIF for confirmation
5. Statsmodels
6. Some skewness in the dataset but that just represents the nature of the dataset -> outlier detection using boxplot
"""


# Working on the Conclusions:
# Drop time col
df.drop('time', axis=1, inplace=True)

# Outlier Detection and Handling
'''
For outlier detection, 
First we plot the box plot for our data
Then we observe the outliers present in the distribution
Then we check in which quantile, clusters of outliers lie the most and based on that-
1. Select a row at particular quantile(.95)
2. Reformat the df by selecting records less than or more than that quantile row thus removing outliers lying above or below the fence or lying in the tail region.
3. Then again plot the box plot to observe the change in the outliers
NOTE: We will definitely lose data as we are discarding data lying in the tail region.
ADV: Fixes skewness in data distribution thus outlier removal at a certain extent.
'''
scaler = StandardScaler()
std_x = scaler.fit_transform(df.drop('label', axis=1))

df_new = df
# Plotting Box Plot after scaling Features values to detect outliers
'''
scaler = StandardScaler()
std_x = scaler.fit_transform(df_new.drop('label', axis=1))
fig, ax = plt.subplots(figsize=(20,20))
sns.boxplot(data=std_x, ax=ax)
'''

q = df_new['avg_rss12'].quantile(0.006)
df_new = df_new[df_new['avg_rss12'] > q]

# Plotting Box Plot after scaling Features values to detect outliers
"""
scaler = StandardScaler()
std_x = scaler.fit_transform(df_new.drop('label', axis=1))
fig, ax = plt.subplots(figsize=(20,20))
sns.boxplot(data=std_x, ax=ax)
"""

q = df_new['var_rss12'].quantile(0.98)
df_new = df_new[df_new['var_rss12'] < q]

# Plotting Box Plot after scaling Features values to detect outliers
"""
scaler = StandardScaler()
std_x = scaler.fit_transform(df_new.drop('label', axis=1))
fig, ax = plt.subplots(figsize=(20,20))
sns.boxplot(data=std_x, ax=ax)
"""

q = df_new['var_rss13'].quantile(0.98)
df_new = df_new[df_new['var_rss13'] < q]

# Plotting Box Plot after scaling Features values to detect outliers
"""
scaler = StandardScaler()
std_x = scaler.fit_transform(df_new.drop('label', axis=1))
fig, ax = plt.subplots(figsize=(20,20))
sns.boxplot(data=std_x, ax=ax)
"""

q = df_new['var_rss23'].quantile(0.99)
df_new = df_new[df_new['var_rss23'] < q]

# Plotting Box Plot after scaling Features values to detect outliers
"""
scaler = StandardScaler()
std_x = scaler.fit_transform(df_new.drop('label', axis=1))
fig, ax = plt.subplots(figsize=(20,20))
sns.boxplot(data=std_x, ax=ax)
"""

q = df_new['var_rss12'].quantile(0.99)
df_new = df_new[df_new['var_rss12'] < q]

# Plotting Box Plot after scaling Features values to detect outliers
"""
scaler = StandardScaler()
std_x = scaler.fit_transform(df_new.drop('label', axis=1))
fig, ax = plt.subplots(figsize=(20,20))
sns.boxplot(data=std_x, ax=ax)
"""

q = df_new['var_rss12'].quantile(0.98)
df_new = df_new[df_new['var_rss12'] < q]

# Plotting Box Plot after scaling Features values to detect outliers
"""
scaler = StandardScaler()
std_x = scaler.fit_transform(df_new.drop('label', axis=1))
fig, ax = plt.subplots(figsize=(20,20))
sns.boxplot(data=std_x, ax=ax)
"""

# Saving Final graph of our dataset(after working on Outliers) distribution using Box Plot
plt.savefig('data_plot')

# Printing the new df
print(df_new)


# Feature and Label selection
y = df_new['label']
x = df_new.drop('label', axis=1)


# Variance Inflation Factor (VIF) to understand the multicollinearity.
vif = pd.DataFrame()
vif['Features'] = x.columns
vif['VIF'] = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
print(vif)
'''
CONCLUSION: No certainly high multicollinearity observed in the data.
'''


# Standard Scaling to scale down each feature on a similar scale
scaler = StandardScaler()
std_x = scaler.fit_transform(x)
print(std_x)


# Saving Scaler model object
pickle.dump(scaler, open('scaler.pickle', 'wb'))


# VIF after scaling
vif_new = pd.DataFrame()
vif_new['Features'] = x.columns
vif_new['VIF'] = [variance_inflation_factor(std_x, i) for i in range(x.shape[1])]
print(vif_new)
'''
CONCLUSION: No certainly high multicollinearity observed in the data.
'''


# Saving the final dataset after preprocessing it and working on the outliers
df_new.to_csv("final_dataset.csv")
