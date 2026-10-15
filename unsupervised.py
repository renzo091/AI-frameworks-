import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("./datasets/Country-data.csv")

print(data.head())
print(data.tail())
print(data.info())


col = list(data.columns)
col.remove('country')
categorical_features = ['country']
numerical_features = [*col]   #creates a list called numerical_features and assigns it 
# the remaining column names from the col list. The * operator is used to unpack the elements 
# in the col list into the numerical_features list. 

print('Categorical Features :',*categorical_features)
print('Numerical Features :',*numerical_features)

fig, ax = plt.subplots(nrows = 3,ncols = 3,figsize = (15,15))
colors = ['#461c00','#ffbf95']
for i in range(len(numerical_features)):
    plt.subplot(3,3,i+1)
    sns.distplot(data[numerical_features[i]],color = colors[0])
    title = 'Distribution : ' + numerical_features[i]
    plt.title(title)
    plt.show()

    #Here is a code example for you

# Add what is needed (y, by and the title)
"""
fig  = plt.subplots(nrows = 3,ncols = 3,figsize = (15,15))
for i in range(len(numerical_features)):
    plt.subplot(3,3,i+1)
    ax = sns.boxplot(data[numerical_features[i]],color = colors[1])
    plt.title(numerical_features[i])
plt.show()
"""

#High
fig = plt.subplots(nrows = 1,ncols = 3,figsize = (25,7))
plt.subplot(1,3,1)
ax = sns.barplot(x = 'country',y ='' , data = data.sort_values(ascending = False,by = 'exports').iloc[:5],palette = colors,edgecolor = 'black')
plt.title()
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(), int(rect.get_height()), 
            horizontalalignment='center', fontsize = 12)

#Medium
plt.subplot(1,3,2)
ax = sns.barplot(x = 'country',y ='' , data = data.sort_values(ascending = False,by = 'exports').iloc[81:86],palette = colors,edgecolor = 'black')
plt.title()
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(), int(rect.get_height()), 
            horizontalalignment='center', fontsize = 12)

#Low
plt.subplot(1,3,3)
ax = sns.barplot(x = 'country',y ='' , data = data.sort_values(ascending = False,by = 'exports').iloc[161:166],palette = colors,edgecolor = 'black')
plt.title()
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(), int(rect.get_height()), 
            horizontalalignment='center', fontsize = 12)
plt.show()