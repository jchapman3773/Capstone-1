import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import XyScaler
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats import outliers_influence, diagnostic
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
df = pd.read_csv('../data/PIRUS.csv',na_values=['-99'])


class Data:

    def __init__(self,data,predict):
        self.df = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predict = predict

    def clean_data(self):
        columns = list(self.df.loc[:,'Broad_Ethnicity':'Aspirations'].columns)
        columns += ['Violent']
        columns.remove('Age_Child')
        self.df.Gender.replace({1:0,2:1})
        self.df.Language_English.replace(-88,1)
        self.df[['Education_Change','Change_Performance','Work_History','Social_Stratum_Adulthood']].replace(-88,'NaN')
        clean_df = self.df[columns].select_dtypes(exclude='object').fillna(df.mean())
        return clean_df

    def data_clean_split(self):
        clean_df = self.clean_data()
        y = clean_df[self.predict]
        X = clean_df.drop(self.predict,axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1)
        return

class Model(Data):

    def __init__(self,data,model,predict):
        self.model = model
        self.trained_model = None
        self.scaler = XyScaler()
        super().__init__(data,predict)

    def get_vifs(self):
        for idx, col in enumerate(self.X_train.columns):
            print(f"{col}: {outliers_influence.variance_inflation_factor(self.X_train.values,idx)}")
        return

    def get_goldsfeltquandt(self):
        print(diagnostic.het_goldfeldquandt(self.trained_model.resid, self.trained_model.model.exog))

    def get_linear_summary(self):
        linear_model = sm.OLS(self.y_train, self.X_train).fit()
        print(linear_model.summary2())


if __name__ == '__main__':
    PIRUS = Model(df, LassoCV(), 'Violent')
    PIRUS.data_clean_split()
    PIRUS.get_vifs()
    PIRUS.get_linear_summary()
