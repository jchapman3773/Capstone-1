import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import XyScaler
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split, KFold
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
        self.predict = predict
        self.X = None
        self.y = None
        self.X_scale = None
        self.y_scale = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.clean_data = None

    def create_clean_data(self):
        columns = list(self.df.loc[:,'Broad_Ethnicity':'Aspirations'].columns)
        columns += ['Violent']
        columns.remove('Age_Child')
        self.df.Gender.replace({1:0,2:1})
        self.df.Language_English.replace(-88,1)
        self.df[['Education_Change','Change_Performance','Work_History','Social_Stratum_Adulthood']].replace(-88,'NaN')
        self.clean_data = self.df[columns].select_dtypes(exclude='object').fillna(df.mean())
        self.y = self.clean_data[self.predict]
        self.X = self.clean_data.drop(self.predict,axis=1)

    def fix_imbalance(self):
        pass

    def scale_data(self):
        X_scale,y_scale = self.scaler.fit_transform(self.X,self.y)
        self.X_scale = pd.DataFrame(data=X_scale,columns=self.X.columns,index=self.X.index)
        self.y_scale = pd.Series(data=y_scale)

    def split_data(self,split=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scale, self.y_scale, test_size=split)

    def prep_data(self):
        self.create_clean_data()
        self.fix_imbalance()
        self.scale_data()
        self.split_data()

class Model(Data):

    def __init__(self,data,model,predict,name):
        super().__init__(data,predict)
        self.model = model
        self.name = name
        self.alphas = None
        self.log_alphas = None
        self.scaler = XyScaler()

    def get_alphas(self):
        self.alphas = self.model.alphas_
        self.log_alphas = np.log10(self.alphas)

    def fit_model(self):
        self.model.fit(self.X_train,self.y_train)

    def plot_coef_log_alphas(self):
        coeffs = self.model.path(self.X_train,self.y_train)[1]
        plt.plot(self.log_alphas,coeffs.T)
        plt.title(f'Coefficient Descent of {self.name}')
        plt.xlabel(r'log($\alpha$)')
        plt.ylabel('Coefficients')
        plt.savefig(f'Coefficient Descent of {self.name}.png')

    def plot_mse(self):
        pass

    def plot_ROC(self):
        pass

    def get_vifs(self):
        for idx, col in enumerate(self.X_train.columns):
            print(f"{col}: {outliers_influence.variance_inflation_factor(self.X_train.values,idx)}")

    def get_goldsfeltquandt(self):
        print(diagnostic.het_goldfeldquandt(self.trained_model.resid, self.trained_model.model.exog))

    def get_linear_summary(self):
        linear_model = sm.OLS(self.y_train, self.X_train).fit()
        print(linear_model.summary2())

    def clean_split_fit(self):
        self.prep_data()
        self.fit_model()
        self.get_alphas()

if __name__ == '__main__':
    PIRUS = Model(df, LassoCV(), 'Violent','Lasso')
    PIRUS.clean_split_fit()
    PIRUS.plot_coef_log_alphas()
    # PIRUS.get_vifs()
    plt.show()
