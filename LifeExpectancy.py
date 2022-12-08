#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:27:02 2020

@author: tanyaedwards
"""

import pandas as pd
import plotly.express as px
import numpy as np
import math
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(11.7,8.27)})
# =============================================================================
# a4_dims = (11.7, 8.27)
# fig, ax = plt.subplots(figsize=a4_dims)
# =============================================================================
plt.ion()



LifeE = pd.read_csv('LifeExpectancy.csv')
LifeE = LifeE.rename(columns={'Life expectancy ':'Life Expectancy',
                              'percentage expenditure':'Percentage Expenditure',
                              'Measles ':'Measles',
                              ' BMI ':'BMI',
                              'under-five deaths ':'under-five deaths',
                              'Total expenditure':'Total Expenditure',
                              'Diphtheria ':'Diphtheria',
                              ' HIV/AIDS':'HIV/AIDS',
                              ' thinness  1-19 years':'thinness 1-19 years',
                              ' thinness 5-9 years':'thinness 5-9 years'})
                     
print(LifeE.columns)
print(LifeE.head())
print(LifeE.info())

#Visualising and Exploring Data
#Fitting three independent variables
r2_df = pd.DataFrame(columns=['Variable', 'r2'])
i=0
dep_var = ['Schooling','Income composition of resources','HIV/AIDS'] #'Diphtheria', 'BMI', 'Polio'
 
for col in dep_var:
    df = LifeE[['Year', 'Country', 'Life Expectancy', col]]
    df = df.dropna(axis=0, how='any').reset_index()  
    df = df[df[col] != 0]
    df['Life Expectancy log'] = np.log(df['Life Expectancy'])
    r2_df.at[i,'Variable']=col
    
    name_y = 'Life Expectancy log'
    name_x = col

    if col == 'HIV/AIDS':
        df[col+' log'] = np.log(df[col])           
        #name_y = 'Life Expectancy log'
        name_x = col+' log'

    X = df[[name_x]]
    y = df[[name_y]]
    Xy = df[[name_y, name_x]] 
    
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    r2 = regr.score(X, y)
    r2_df.at[i,'r2']=r2
    
    y_pred = regr.predict(X)
    df['y_pred'] = y_pred[:,0]
    
    #Plotly
    fig1 = px.scatter(Xy, x=name_x, y=name_y) 
    fig1.add_scatter(x=X[name_x], y=df['y_pred']) 
    if col == 'HIV/AIDS':
        col = 'HIV_AIDS'
    fig1.write_image('Figures/Plotly/LifeExpectancy_{}.jpg'.format(col))
    
    #Matplotlib
    plt.figure()
    plt.plot(Xy[name_x], Xy[name_y], "o", alpha=0.4)
    plt.savefig('Figures/Matplotlib/LifeExpectancy_{}.jpg'.format(col))
    plt.close()
        
    #Seaborn
    sns_plot = sns.lmplot(data=Xy, x=name_x, y=name_y, line_kws={'color': 'orange'}, 
                          scatter_kws={'color': '#1f77b4', 'alpha':0.4},
                          height=8.27, aspect=11.7/8.27)
    sns_plot.savefig('Figures/Seaborn/LifeExpectancy_{}.jpg'.format(col))
    plt.show()
    plt.close()
          
    i+=1


#----------3D Plots-------------------
scatter_df = LifeE[['Year', 'Country', 'Life Expectancy', 'Schooling',
                   'Income composition of resources','HIV/AIDS']]
scatter_df = scatter_df.dropna(axis=0, how='any').reset_index()
scatter_df['Life Expectancy log'] = np.log(scatter_df['Life Expectancy'])
scatter_df['HIV/AIDS log'] = np.log(scatter_df['HIV/AIDS'])

fig2 = px.scatter_3d(scatter_df, x='HIV/AIDS log', y='Income composition of resources',
                     z='Life Expectancy log')
fig2.write_image('Figures/LifeExpectancy_variables_3D.jpg')

#scatter 3D Matplotlib
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(scatter_df['Schooling'], scatter_df['HIV/AIDS log'], 
           scatter_df['Life Expectancy log'], c=scatter_df['Life Expectancy log'], 
           cmap='viridis', linewidth=0.5)
#ax.plot_trisurf(scatter_df['Schooling'], scatter_df['Income composition of resources'], 
#           scatter_df['Life Expectancy log'], cmap='viridis', edgecolor='none')
plt.xlabel('Schooling')
plt.ylabel('Log(HIV/AIDS)')
ax.set_zlabel('log(Life Expectancy)', rotation=90) 
#plt.zlabel('Logarithm of Life Expectancy')
plt.title('Target against Feature Variables')
plt.savefig('Figures/3DSurfacePlot.jpg')
plt.show()

# =============================================================================
# #surface plane plot (ideal)
# plt.figure()
# ax = plt.axes(projection='3d')
# scatter_points_df = pd.DataFrame(columns=['x', 'y', 'z'])
# #for col in scatter_points_df.columns.tolist():
# #    scatter_points_df[col] = range(1,1000)
# scatter_points_df['x'] = range(1,1000)
# scatter_points_df['y'] = range(1, 1000)
# scatter_points_df['z'] = range(1,1000)
# ax.scatter(scatter_points_df['x'], scatter_points_df['y'], scatter_points_df['z'],
#            c=scatter_points_df['z'], 
#            cmap='viridis', linewidth=0.5)
# =============================================================================



#---------MLR Model-----------------
#sel_cols = ['Life Expectancy'] + LifeE.columns.tolist()[4:22]
sel_cols = ['Life Expectancy','Schooling','Income composition of resources',
            'HIV/AIDS'] #, 'Diphtheria', 'BMI', 'Polio'
model_df_x = LifeE[sel_cols]
model_df_x = model_df_x.dropna(axis=0, how='any').reset_index(drop=True)
model_df_x['Life Expectancy log'] = np.log(model_df_x['Life Expectancy'])
model_df_x['HIV/AIDS log'] = np.log(model_df_x['HIV/AIDS'])
model_df_y = model_df_x[['Life Expectancy log']]
model_df_x = model_df_x.drop(['Life Expectancy', 'Life Expectancy log', 'HIV/AIDS'], axis=1)
    
x_train, x_test, y_train, y_test = train_test_split(model_df_x, model_df_y, 
                                                    train_size=0.8, test_size=0.2)

mlr = linear_model.LinearRegression()
model = mlr.fit(x_train, y_train)

#first use .score() to tweak model and find best paramters (best R2)
#can use subsets of train data to tweak
score_train = mlr.score(x_train, y_train) 
print('Train score: {}'.format(score_train))

#getting p-values for variables
est = sm.OLS(y_train, sm.add_constant(x_train))
est2 = est.fit()
print(est2.summary())
p_values = est2.summary2().tables[1]['P>|t|']
print(p_values)

#-------------
#find predicted values on test data and find final R2 value from test data
y_predict = mlr.predict(x_test)
y_predict_df = pd.DataFrame()
y_predict_df['Life Expectancy log (observed)'] = y_test['Life Expectancy log']
y_predict_df['Life Expectancy log (predicted)'] = y_predict[:,0]
y_predict_df['Life Expectancy (obeserved)'] = (math.e)**y_predict_df['Life Expectancy log (observed)']
y_predict_df['Life Expectancy (predicted)'] = (math.e)**y_predict_df['Life Expectancy log (predicted)']

plt.figure()
plt.plot(y_test, y_predict, "o", alpha=0.4)
plt.title('Predicted vs True Values')
plt.xlabel('Predicted Value of log(Life Expectancy)')
plt.ylabel('True Value of log(Life Expectancy')
plt.show()

xy = pd.DataFrame()
xy['y_test'] = y_test['Life Expectancy log']
xy['y_predict'] = y_predict[:,0]
sns.scatterplot(data=xy, x='y_predict', y='y_test', color='#1f77b4', alpha=0.4)
                          
plt.show()
#--------------

residuals = y_predict - y_test
plt.figure()
plt.scatter(y_predict[:,0], residuals, alpha=0.4)
plt.ylim(-0.165, 0.165)
plt.xlim(3.75, 4.5)
plt.title('Residual Analysis')
plt.xlabel('Predicted Value of log(Life Expectancy)')
plt.ylabel('Residual')
plt.show()

#print(type(y_predict[:,0].tolist()))
residuals_df = pd.DataFrame(data={'y_predict':y_predict[:,0].tolist(), #y_predict_df['Life Expectancy log'],
                                  'residuals':residuals['Life Expectancy log']})
plot = sns.scatterplot(data=residuals_df, x='y_predict', y='residuals', color='#1f77b4', alpha=0.4)
plt.show()


#find r2 from unbiased test dataset
score_test = mlr.score(x_test, y_test)
print('Test score: {}'.format(score_test))

#Predict Y value for certain X value
df_query = pd.DataFrame(columns=sel_cols[1:])
df_query.loc[0] = [10.0, 0.5, 2.5]
y_query = mlr.predict(df_query)
life_expectancy_query = math.e**y_query[0,0]
print(life_expectancy_query)


#-----------code check-----------------------------
#checking r2 values for scikit-learn and statsmodels when fitting and not fitting intercept
sklernIntercept=linear_model.LinearRegression(fit_intercept=True).fit(x_train, y_train)
sklernNoIntercept=linear_model.LinearRegression(fit_intercept=False).fit(x_train, y_train)
statsmodelsIntercept = sm.OLS(y_train, sm.add_constant(x_train))
statsmodelsNoIntercept = sm.OLS(y_train, x_train)
#print(sklernIntercept.score(x_train, y_train), statsmodelsIntercept.fit().rsquared)
#print(sklernNoIntercept.score(x_train, y_train), statsmodelsNoIntercept.fit().rsquared)


# =============================================================================
# r2_df = pd.DataFrame(columns=['Variable', 'r2'])
# i=0
# for col in dep_var:
#     r2 = LinearFit(col, LifeE, display=False)
#     r2_df.at[i,'Variable']=col
#     r2_df.at[i,'r2']=r2
#     i+=1
# r2_df.head()
# =============================================================================

#--------------Initially Exploring Data-----------------------------------------
#Checking Fits of all variables
r2_df_all = pd.DataFrame(columns=['Variable', 'r2', 'r2_log'])
i=0
for col in LifeE.columns.tolist()[15:16]:
    df = LifeE[['Year', 'Country', 'Life Expectancy', col]]
    df = df.dropna(axis=0, how='any').reset_index()    
    #df = df[df['Year']==2015]
    r2_df_all.at[i,'Variable']=col

    #if col == col: #'HIV/AIDS'
    df = df[df[col] != 0]
    df[col+' log'] = np.log(df[col])  
    df['Life Expectancy log'] = np.log(df['Life Expectancy'])

    X = df[[col+' log']] ##
    y = df[['Life Expectancy log']]
    Xy = df[['Life Expectancy log', col+' log']] ##
    
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    r2 = regr.score(X, y)
    r2_df_all.at[i,'r2_log']=r2
    
    y_pred = regr.predict(X)
    df['y_pred'] = y_pred[:,0]
    fig1 = px.scatter(Xy, x=col+' log', y='Life Expectancy log') ##
    fig1.add_scatter(x=X[col+' log'], y=df['y_pred']) ## +' log'
    #fig1.show(renderer='browser')
          
    i+=1

i=0
for col in LifeE.columns.tolist()[15:16]:
    df = LifeE[['Year', 'Country', 'Life Expectancy', col]]
    df = df.dropna(axis=0, how='any').reset_index()    
    df = df[df[col] != 0]

    X = df[[col]]
    y = df[['Life Expectancy']]
    Xy = df[['Life Expectancy', col]]
    
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    r2 = regr.score(X, y)
    
    y_pred = regr.predict(X)
    df['y_pred'] = y_pred[:,0]
    fig1 = px.scatter(Xy, x=col, y='Life Expectancy')
    fig1.add_scatter(x=X[col], y=df['y_pred'])
    #fig1.show(renderer='browser')
    
    r2_df_all.at[i,'r2']=r2
    i+=1
    
    
    
    
    
#-----------------------------

#Alcohol
Alcohol = LifeE[['Year', 'Country', 'Life Expectancy', 'Alcohol']]
Alcohol = Alcohol.dropna(axis=0, how='any').reset_index()
#Alcohol = Alcohol[Alcohol['Year']==2015]

Expenditure = LifeE[['Year', 'Country', 'Life Expectancy', 'Percentage Expenditure']]
Expenditure = Expenditure.dropna(axis=0, how='any').reset_index()
#Expenditure = Expenditure[Expenditure['Year']==2015]

#Linear Fit
X_alcohol = Alcohol[['Alcohol']]
y_alcohol = Alcohol[['Life Expectancy']]
regr_alcohol = linear_model.LinearRegression()
regr_alcohol.fit(X_alcohol, y_alcohol)
r2_alcohol = regr_alcohol.score(X_alcohol, y_alcohol)
#print(r2_alcohol)

X_expenditure = Expenditure[['Percentage Expenditure']]
y_expenditure = Expenditure[['Life Expectancy']]
regr_expenditure = linear_model.LinearRegression()
regr_expenditure.fit(X_expenditure, y_expenditure)
r2_expenditure = regr_expenditure.score(X_expenditure, y_expenditure)
#print(r2_expenditure)


fig1 = px.scatter(Alcohol, x='Alcohol', y='Life Expectancy')
#fig1.show(renderer='browser')
fig2 = px.scatter(Expenditure, x='Percentage Expenditure', y='Life Expectancy')
#fig2.show(renderer='browser')

