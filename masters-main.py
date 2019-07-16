# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 21:09:56 2019

@author: Larsmartin
"""

import pandas as pd 
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.metrics import mean_absolute_error
from datetime import date
import datetime

#%%
#setting the different features used for the machine learning, mainly spectral values from seperate dates


#features = ['02-07-18_greenmedian','05.07.18_redmedian', '14-07-17_ndvimedian', '14-07-17_nirmedian', '14-07-17_rededgemedian', '14-07-17_mtci', '17-07-17_greenmedian','17-07-17_redmedian', '17-07-17_ndvimedian', '17-07-17_nirmedian', '17-07-17_rededgemedian', '17-07-17_mtci', '20-07-17_greenmedian','20-07-17_redmedian', '20-07-17_ndvimedian', '20-07-17_nirmedian', '20-07-17_rededgemedian', '20-07-17_mtci']
#features =['14-07-17_redmedian', '14-07-17_ndvimedian', '17-07-17_greenmedian','20-07-17_redmedian', '20-07-17_mtci']
#features = ['02-07-18_ndvimedian', '02-07-18_rededgemedian', '05.07.18_bluemedian',
#       '05.07.18_redmedian', '05.07.18_ndvimedian', '05.07.18_nirmedian',
#       '11.07.18_bluemedian', '11.07.18_greenmedian', '11.07.18_redmedian',
#       '11.07.18_rededgemedian', '19.07.18_ndvimedian', '19.07.18_mtci',
#       '24.07.18_bluemedian']
features = ['26-06-18_bluemedian', '26-06-18_redmedian', '26-06-18_mtci', 'MAT',	
       '02-07-18_mtci', '02-07-18_evi', '19-07-18_bluemedian',	
       '19-07-18_greenmedian', '19-07-18_redmedian', '19-07-18_nirmedian',	
       '19-07-18_mtci']
#features = ['02-07-18_bluemedian', '02-07-18_greenmedian','02-07-18_redmedian', '02-07-18_ndvimedian', 
#            '02-07-18_nirmedian', '02-07-18_rededgemedian', '02-07-18_mtci', '02-07-18_evi','05.07.18_bluemedian',
#            '05.07.18_greenmedian','05.07.18_redmedian', '05.07.18_ndvimedian', '05.07.18_nirmedian', 
#            '05.07.18_rededgemedian', '05.07.18_mtci', '05.07.18_evi', '11.07.18_bluemedian', 
#            '11.07.18_greenmedian','11.07.18_redmedian', '11.07.18_ndvimedian', '11.07.18_nirmedian', 
#            '11.07.18_rededgemedian', '11.07.18_mtci', '11.07.18_evi','19.07.18_bluemedian', '19.07.18_greenmedian',
#            '19.07.18_redmedian', '19.07.18_ndvimedian', '19.07.18_nirmedian', '19.07.18_rededgemedian', 
#            '19.07.18_mtci', '19.07.18_evi','24.07.18_bluemedian', '24.07.18_greenmedian','24.07.18_redmedian', 
#            '24.07.18_ndvimedian', '24.07.18_nirmedian', '24.07.18_rededgemedian', '24.07.18_mtci', '24.07.18_evi', 
#            'MAT', '05.07.18_mtci - 02-07-18_mtci', '11.07.18_mtci - 02-07-18_mtci',
#       '19.07.18_mtci - 02-07-18_mtci', '24.07.18_mtci - 02-07-18_mtci']



#%%importing excel file, setting index in dataframe, removing rows with nan-values
dfferdig= pd.read_excel('filename.xlsx')

dfferdig.set_index('26-06-18Unnamed: 0', inplace=True)
dfferdig.dropna(axis=0, subset=['26-06-18GrainYield'], inplace=True)

#%%
#setting a feature of differences in mtci 
#delta = (dfferdig['13-07-18MAT'].max() - datetime.datetime(2018,7,1))
##delta = dfferdig['13-07-18MAT'].max()-dfferdig['13-07-18MAT'].min()
#delta.days
datelist = []
for dato in dfferdig['13-07-18MAT']:
    delta = dato - datetime.datetime(2018,7,1)
    datelist.append(delta.days+1)
    

dfferdig['13-07-18MAT'] = datelist 
dfferdig.rename({'13-07-18MAT':'MAT'}, inplace = True, axis='columns')

#%%
#newest features, the ones gotten from sfs
c17 = ['14-06-17_ndvimedian', '29-06-17_greenmedian', '29-06-17_nirmedian',					
       '03-07-17_greenmedian', '03-07-17_rededgemedian',					
       '17-07-17_rededgemedian', '14-08-17_ndvimedian', '14-08-17_nirmedian',					
       '14-08-17_rededgemedian', '14-08-17_mtci',					
       '03-07-17_mtci - 14-06-17_mtci']					
a17 = ['17-07-17_greenmedian', '20-07-17_redmedian', '20-07-17_mtci',				
       '01-08-17_ndvimedian']
c18 = ['02-07-18_greenmedian', '02-07-18_ndvimedian', '02-07-18_evi',				
       '05.07.18_bluemedian', '05.07.18_mtci', '11.07.18_bluemedian',				
       '11.07.18_greenmedian', '11.07.18_rededgemedian', '19.07.18_redmedian',				
       '19.07.18_mtci', '24.07.18_ndvimedian', '24.07.18_evi']
b18=['13-07-18_bluemedian', '13-07-18_greenmedian', '13-07-18_ndvimedian',		
       '13-07-18_rededgemedian', '13-07-18_evi', '26-07-18_ndvimedian',		
       '26-07-18_mtci']
a18 = ['26-06-18_bluemedian', '26-06-18_redmedian', '26-06-18_mtci',	
       '02-07-18_mtci', '02-07-18_evi', '19-07-18_bluemedian',	
       '19-07-18_greenmedian', '19-07-18_redmedian', '19-07-18_nirmedian',	
       '19-07-18_mtci']
#checking if MAT in list
for liste in [c17, a17, c18, b18, a18]:
    if 'MAT' in liste :
        print('ja')
    else:
        print('nei')

#%%creating dataframes with correct x and y values, scaling

df_x = dfferdig[features]
#df_x['mtci-14-07 minus 14-06'] = df_x['14-07-17_mtci'] - df_x['14-06-17_mtci']
df_y = dfferdig['26-06-18GrainYield']
#df_y.isna().sum()
df_y.describe()
scaler= StandardScaler()
df_x.iloc[:, :] = scaler.fit_transform(df_x.iloc[:, :].values)


#%%setting correct c-value from the SVR algorithm
liste_train =[]
x_ax= np.linspace(0,1.1, 9)
liste_test =[]
rtest2=[]
rtrain2 =[]
abs_err =[]
clist= [1, 50, 100, 120,140, 150,170, 190, 200,220, 250, 270, 300, 400, 500]
abserr2 =[]
for c in clist:
    for rand in range(1,101):
        X_train, X_test, y_train, y_test = train_test_split(
            df_x, df_y, test_size=0.3, random_state=rand)
        
        svr =SVR(kernel='rbf', C=c, epsilon=0.1, verbose=True, gamma='auto')
        svr.fit(X_train, y_train)
        liste_test.append(svr.score(X_test, y_test))
        liste_train.append(svr.score(X_train, y_train))
        abs_err.append(mean_absolute_error(y_test, svr.predict(X_test)))
    abserr2.append(np.mean(abs_err))
    rtest2.append(np.mean(liste_test))
    rtrain2.append(np.mean(liste_train))
#print(np.mean(liste_test))    
#print(np.mean(liste_train))
#print(np.mean(abs_err))
np.max(rtest2)

pred = svr.predict(X_test)
#plt.plot(clist, rtest2,'r', clist, rtrain2,'b')

line1, = plt.plot(clist, rtest2, label="Test", linestyle='--')
line2, = plt.plot(clist, rtrain2, label="Train", linewidth=4)

first_legend = plt.legend(handles=[line1], loc=4)

# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)

# Create another legend for the second line.
plt.legend(handles=[line2], loc=5)
plt.xlabel('C-values')
plt.ylabel('R^2')
plt.show()
#plt.plot(clist, line1,'r', clist, line2,'b')

#b =y_test.values.ravel()
#svr.predict(X_test)
#plt.scatter(pred, y_test)
#plt.xlabel('Predicted grain yield')
#plt.ylabel('Measured grain yield')
#%%training and testing with correct c and set of features using SVR
liste_train =[]
x_ax= np.linspace(0,1.1, 9)
liste_test =[]
rtest2=[]
rtrain2 =[]
abs_err =[]
abserr2 =[]
for rand in range(1,101):
    X_train, X_test, y_train, y_test = train_test_split(
        df_x, df_y, test_size=0.3, random_state=rand)
    
    svr =SVR(kernel='rbf', C=350, epsilon=0.1, verbose=True, gamma='auto')
    svr.fit(X_train, y_train)
    liste_test.append(svr.score(X_test, y_test))
    liste_train.append(svr.score(X_train, y_train))
    abs_err.append(mean_absolute_error(y_test, svr.predict(X_test)))
    
print(np.mean(liste_test))    
#print(np.mean(liste_train))
print(np.mean(abs_err))


pred = svr.predict(X_test)
#b =y_test.values.ravel()
#svr.predict(X_test)
plt.scatter(pred, y_test)
plt.xlabel('Predicted grain yield')
plt.ylabel('Measured grain yield')
plt.plot(y_test, y_test, '-',color='red')
#%%sfs, finding correct features using sequential feature selector and SVR
svr =SVR(kernel='rbf', C=300, epsilon=0.1, verbose=2, gamma='auto')
sfs = SFS(svr, 
          k_features=27, 
          forward=True, 
          floating=True, 
          verbose=1,
          scoring='neg_mean_absolute_error',
          cv=20,
          n_jobs=3)

sfs1 = sfs.fit(df_x.values,df_y)

#sfs1.subsets_
sfs1.k_feature_names_
res =pd.DataFrame.from_dict(sfs1.get_metric_dict()).T
sfs1.k_score_
fig = plot_sfs(sfs1.get_metric_dict(), kind='std_err')
print(df_x.columns[[sfs1.k_feature_idx_]])

plt.title('Sequential Forwards Selection (w. StdErr)')
plt.ylabel('neg mean absolute error')
plt.grid()
plt.show()

#%%
#finding features that were the most important in the outout of the sfs-algorithm. check the output from "res" and set 
#the most important indices in "important"
colliste= df_x.columns
important = ['2', '12', '15', '18', '22', '34', '38', '39', '40', '41', '44']
imp_feat = colliste[[int(ind) for ind in important]]
print(imp_feat)

