import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import matplotlib as mpl
mpl.rcParams.update({
    'figure.figsize'      : (15,15),
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'large',
    'legend.loc'          : 'upper right'
})

# Import Data with -99 as NaN
df = pd.read_csv('PIRUS.csv',na_values=['-99'])

# Select and clean Data and split for X,y
columns = list(df.loc[:,'Broad_Ethnicity':'Aspirations'].columns)
columns.remove('Age_Child')
df.Gender.replace({1:0,2:1})
df.Language_English.replace(-88,1)
df[['Education_Change','Change_Performance','Work_History','Social_Stratum_Adulthood']].replace(-88,'NaN')
y = df['Violent']
X = df[columns].select_dtypes(exclude='object').fillna(df.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

Class Model():

    __init__(self):
    pass

def get_vif():
    pass

def clean_data():
    pass
