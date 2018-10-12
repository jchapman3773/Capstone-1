import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fancyimpute import SimpleFill, KNN, IterativeSVD, MatrixFactorization
from imblearn.over_sampling import SMOTE
from utils import XyScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
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

class Data:

    def __init__(self,data,predict):
        self.df = data
        self.predict = predict
        self.X = None
        self.y = None
        self.X_scale = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.incomplete_data = None
        self.clean_data = None
        self.methods = [SimpleFill(), KNN(1), KNN(2), KNN(3), KNN(4), KNN(5), IterativeSVD(), MatrixFactorization()]

    def create_clean_data(self):
        columns = list(self.df.loc[:,'Broad_Ethnicity':'Aspirations'].columns)
        columns += ['Violent','Plot_Target1','Criminal_Severity','Current_Status','Group_Membership',
                    'Length_Group','Radical_Behaviors','Radical_Beliefs','Abuse_Child','Psychological',
                    'Alcohol_Drug','Close_Family','Previous_Criminal_Activity','Angry_US']
        columns.remove('Age_Child')
        self.df.Plot_Target1.replace(-88,0,inplace=True)
        self.df.Length_Group.replace(-88,0,inplace=True)
        self.df.Gender.replace({1:0,2:1},inplace=True)
        self.df.Language_English.replace({1:2,-88:1},inplace=True)
        self.df.Education_Change.replace(-88,'NaN',inplace=True)
        self.df.Change_Performance.replace(-88,'NaN',inplace=True)
        self.df.Work_History.replace(-88,'NaN',inplace=True)
        self.df.Social_Stratum_Adulthood.replace(-88,'NaN',inplace=True)
        self.incomplete_data = self.df[columns].select_dtypes(exclude='object')

    def impute(self,method=KNN(5)):
        self.clean_data = pd.DataFrame(data=method.fit_transform(self.incomplete_data),
                        columns=self.incomplete_data.columns,index=self.incomplete_data.index)
        return self.clean_data

    def try_many_imputes(self):
        mse_list = []
        for m in self.methods:
            data = self.impute(m)
            mse_list += [(f'{m.__class__.__name__}',mse(self.incomplete_data.fillna(0),data))]
        with open("../data/impute_mse.txt", "w") as text_file:
            [print(f'{_[0]}, {_[1]}', file=text_file) for _ in mse_list]
        return mse_list

    def make_Xy(self):
        self.y = self.clean_data[self.predict]
        self.X = self.clean_data.drop(self.predict,axis=1)

    def scale_data(self):
        X_scale, _ = self.scaler.fit_transform(self.X,self.y)
        self.X_scale = pd.DataFrame(data=X_scale,columns=self.X.columns,index=self.X.index)

    def fix_imbalance(self,method=SMOTE()):
        os = method
        self.X_scale,self.y = os.fit_sample(self.X_scale,self.y)

    def split_data(self,split=0.25):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scale, self.y,
                                                                    test_size=split, random_state=0)

    def prep_data(self,impute=KNN(5)):
        self.create_clean_data()
        self.impute(impute)
        self.make_Xy()
        self.scale_data()
        if set(self.y) == {0,1}:
            self.fix_imbalance()
        self.split_data()

    def make_heatmap(self):
        corr = self.clean_data.corr()
        sns.set(font_scale=1.1)
        sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
        plt.subplots_adjust(left=0.25,bottom=0.3)
        plt.savefig('../plots/Correlation_Heatmap.png')

    def plot_scatter(self,var1,var2):
        plt.scatter(self.df[var1],self.df[var2],alpha=0.5)
        plt.ylabel(var1)
        plt.xlabel(var2)
        plt.title(f'{var1} vs {var2}')
        plt.savefig(f'../plots/{var1}_vs_{var2}')
