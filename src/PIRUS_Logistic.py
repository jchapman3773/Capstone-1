import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import XyScaler
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV, LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from statsmodels.stats import outliers_influence, diagnostic
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
        self.df.Language_English.replace({1:2,-88:1})
        self.df[['Education_Change','Change_Performance','Work_History','Social_Stratum_Adulthood']].replace(-88,'NaN')
        self.clean_data = self.df[columns].select_dtypes(exclude='object').fillna(df.mean())
        self.y = self.clean_data[self.predict]
        self.X = self.clean_data.drop(self.predict,axis=1)

    def fix_imbalance(self):
        pass

    def scale_data(self):
        X_scale,y_scale = self.scaler.fit_transform(self.X,self.y)
        self.X_scale = pd.DataFrame(data=X_scale,columns=self.X.columns,index=self.X.index)
        # self.y_scale = pd.Series(data=y_scale)

    def split_data(self,split=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scale, self.y, test_size=split)

    def prep_data(self):
        self.create_clean_data()
        self.fix_imbalance()
        self.scale_data()
        self.split_data()

    def make_heatmap(self):
        corr = self.clean_data.corr()
        sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
        plt.subplots_adjust(left=0.25,bottom=0.3)
        plt.savefig('Correlation_Heatmap.png')

    def plot_scatter(self,var1,var2):
        plt.scatter(var1,var2)

class Model(Data):

    def __init__(self,data,model,predict,name):
        super().__init__(data,predict)
        self.model = model
        self.name = name
        self.scaler = XyScaler()

    def fit_model(self):
        self.model.fit(self.X_train,self.y_train)

    def plot_coef_log_alphas(self):
        coeffs = self.model.coefs_paths_.values
        import pdb; pdb.set_trace()
        plt.plot(self.log_alphas,coeffs)
        plt.axvline(np.log10(self.model.alpha_),linestyle='--')
        plt.title(f'Coefficient Descent of {self.name}')
        plt.xlabel('Cs')
        plt.ylabel('Coefficients')
        plt.savefig(f'{self.name}_{self.predict}_coefficient_descent.png')

    def plot_mse(self):
        mse_path = self.model.mse_path_[:,1:]
        mean_mse = mse_path.mean(axis=1)
        plt.plot(self.log_alphas,mse_path,linestyle='--')
        plt.plot(self.log_alphas,mean_mse,label='Mean MSE',linewidth=5,color='k')
        plt.title(r'MSE vs log($\alpha$)')
        plt.xlabel(r'log($\alpha$)')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(f'{self.name}_{self.predict}_MSE_plot.png')

    def plot_scores_kfold(self):
        logistic = LogisticRegression(penalty='l1', solver='saga', random_state=0)
        Cs = np.linspace(0.001, 1000, 30)

        tuned_parameters = [{'C': Cs }]

        clf = GridSearchCV(logistic, tuned_parameters, cv=10, refit=False)
        clf.fit(self.X_train, self.y_train)
        scores = clf.cv_results_['mean_test_score']
        scores_std = clf.cv_results_['std_test_score']
        plt.semilogx(Cs, scores)

        # plot error lines showing +/- std. errors of the scores
        std_error = scores_std / np.sqrt(10)

        plt.semilogx(Cs, scores + std_error, 'b--')
        plt.semilogx(Cs, scores - std_error, 'b--')
        plt.fill_between(Cs, scores + std_error, scores - std_error, alpha=0.2)

        plt.ylabel('CV score +/- std error')
        plt.xlabel('Cs')
        plt.axhline(np.max(scores), linestyle='--', color='.5')
        # plt.xlim([alphas[0], alphas[-1]])
        plt.savefig(f'{self.name}_{self.predict}_kfold_mean_scores.png')

    def plot_ROC(self):
        pass

    def predict_y(self):
        return self.model.predict(self.X_test)

    def print_score(self):
        score = self.model.score(self.X_test,self.y_test)
        print(f"Score: {score}")

    def print_alpha(self):
        print(f"Alpha: {self.model.alpha_}")

    def print_vifs(self):
        for idx, col in enumerate(self.X_train.columns):
            print(f"{col}: {outliers_influence.variance_inflation_factor(self.X_train.values,idx)}")

    def print_coefs(self):
        for idx, col in enumerate(self.X_train.columns):
            print(f"{col}: {self.model.coef_[idx]}")

    def print_goldsfeltquandt(self):
        linear_model = sm.OLS(self.y_train, self.X_train).fit()
        print(diagnostic.het_goldfeldquandt(linear_model.resid, linear_model.model.exog))

    def print_linear_summary(self):
        linear_model = sm.OLS(self.y_train, self.X_train).fit()
        print(linear_model.summary2())

    def clean_split_fit(self):
        self.prep_data()
        self.fit_model()

if __name__ == '__main__':
    df = pd.read_csv('../data/PIRUS.csv',na_values=['-99'])
    PIRUS = Model(df, LogisticRegressionCV(penalty='l1',cv=10,solver='saga'), 'Violent','LogisticRegression')
    PIRUS.clean_split_fit()
    # PIRUS.print_score()
    # PIRUS.plot_coef_log_alphas()
    # PIRUS.plot_mse()
    PIRUS.plot_scores_kfold()
    plt.show()
