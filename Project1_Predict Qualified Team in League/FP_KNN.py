# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import itertools
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import NullFormatter
from sklearn import preprocessing
%matplotlib inline

df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0120ENv3/Dataset/ML0101EN_EDX_skill_up/cbb.csv')
#df.head()

df['WinIndex'] = np.where(df.WAB > 7, 'True', 'False')    #add new col 'WinIndex'; if WAB>7 as 'True', other as 'False'
df1 = df.loc[df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]  #filter dataset to have only qualified teams (entering S16, E8, F4), save to new data frame df1
#df1['POSTSEASON'].value_counts()

import seaborn as sns

bins = np.linspace(df1.BARTHAG.min(), df1.BARTHAG.max(), 10)    #plot for 10 rows/samples within min and max value of BARTHAG
g = sns.FacetGrid(df1, col="WinIndex", hue="POSTSEASON", palette="Set1", col_wrap=6)
g.map(plt.hist, 'BARTHAG', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

bins = np.linspace(df1.ADJOE.min(), df1.ADJOE.max(), 10)
g = sns.FacetGrid(df1, col="WinIndex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJOE', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

bins = np.linspace(df1.ADJDE.min(), df1.ADJDE.max(), 10)
g = sns.FacetGrid(df1, col="WinIndex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJDE', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

#command for check data: df1.groupby(['WinIndex'])['POSTSEASON'].value_counts(normalize=True)
#probability for each season

df1['WinIndex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)    #subs True=1, False=0

#define feature of X from df1
X = df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'WinIndex']]

#store col df1 POSTSEASON to y
y = df1['POSTSEASON'].values

#normalize data for training
X = preprocessing.StandardScaler().fit(X).transform(X)

#split X into train and test to find the best k value
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Validation set:', X_val.shape,  y_val.shape, "\n\n")

#using KNN
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

k = 5       #start algorithm with k=5
nb = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train) #train model from the neighbourhood
ypredict = nb.predict(X_val)

#calculate accuracy train and test
print("Train set Accuracy: ", accuracy_score(y_train, nb.predict(X_train)))
print("Test set Accuracy: ", accuracy_score(y_val, ypredict))

mean_acc = np.zeros((16-1))
std_acc = np.zeros((16-1))

#check accuracy of each value of k
for n in range(1,16):
    
    #Train Model and Predict  
    nb = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    ypredict=nb.predict(X_val)
    mean_acc[n-1] = accuracy_score(y_val, ypredict)
    std_acc[n-1]=np.std(ypredict==y_val)/np.sqrt(ypredict.shape[0])
    print("Train set with k = ", n, "and Accuracy of", accuracy_score(y_train, nb.predict(X_train)))
    print("Test set with k = ", n, " and Accuracy of: ", accuracy_score(y_val, ypredict), "\n")

#plot graph
plt.plot(range(1,16),mean_acc,'g')
plt.fill_between(range(1,16),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,16),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()