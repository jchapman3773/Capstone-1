from init_data import Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import XyScaler
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV, LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_selection import RFE
from statsmodels.stats import outliers_influence, diagnostic
import matplotlib as mpl
mpl.rcParams.update({
    'figure.figsize'      : (15,10),
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'large',
    'legend.loc'          : 'upper right'
})

class LinearModel(Data):

    def __init__(self,data,model,predict,name):
        super().__init__(data,predict)
        self.model = model
        self.name = name
        self.alphas = None
        self.log_alphas = None
        self.scaler = XyScaler()
        self.columns = None

    def select_features(self,features=22):
        selector = RFE(self.model,features)
        self.selector = selector.fit(self.X_train,self.y_train)
        self.X_train = self.X_train.[:,self.selector.support_]
        self.X_test = self.X_test.[:,self.selector.support_]
        self.columns = self.X.loc[:,self.selector.support_].columns

    def get_alphas(self):
        self.alphas = self.model.alphas_
        self.log_alphas = np.log10(self.alphas)

    def fit_model(self):
        self.model.fit(self.X_train,self.y_train)

    def clean_split_fit(self):
        self.prep_data()
        self.select_features()
        self.fit_model()
        self.get_alphas()

    def plot_coef_log_alphas(self):
        coeffs = self.model.path(self.X_train,self.y_train)[1]
        plt.plot(self.log_alphas,coeffs.T)
        plt.axvline(np.log10(self.model.alpha_),linestyle='--')
        plt.title(f'Coefficient Descent of {self.name}')
        plt.xlabel(r'log($\alpha$)')
        plt.ylabel('Coefficients')
        plt.legend(np.append(self.columns.values,['Chosen Alpha']), fontsize = 'x-small',loc='upper left')
        plt.savefig(f'../plots/{self.name}_{self.predict}_coefficient_descent.png')

    def plot_mse(self):
        mse_path = self.model.mse_path_
        mean_mse = mse_path.mean(axis=1)
        plt.plot(self.log_alphas,mse_path,linestyle='--')
        plt.plot(self.log_alphas,mean_mse,label='Mean MSE',linewidth=5,color='k')
        plt.title(r'MSE for kfolds vs log($\alpha$)')
        plt.xlabel(r'log($\alpha$)')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(f'../plots/{self.name}_{self.predict}_MSE_plot.png')

    def plot_scores_kfold(self):
        lasso = Lasso(random_state=0)
        alphas = np.logspace(-4, -0.5, 30)

        tuned_parameters = [{'alpha': alphas}]

        clf = GridSearchCV(lasso, tuned_parameters, cv=10, refit=False)
        clf.fit(self.X_train, self.y_train)
        scores = clf.cv_results_['mean_test_score']
        scores_std = clf.cv_results_['std_test_score']
        plt.semilogx(alphas, scores)

        std_error = scores_std / np.sqrt(10)
        plt.semilogx(alphas, scores + std_error, 'b--')
        plt.semilogx(alphas, scores - std_error, 'b--')
        plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

        plt.title('CV score +/- std error')
        plt.ylabel('Score')
        plt.xlabel(r'$\alpha$')
        plt.axhline(np.max(scores), linestyle='--', color='.5')
        plt.savefig(f'../plots/{self.name}_{self.predict}_kfold_mean_scores.png')

    def print_score(self,test=True):
        if test:
            score = self.model.score(self.X_test,self.y_test)
        else:
            score = self.model.score(self.X_train,self.y_train)
        print(f"{'Test' if test else 'Train'} Score: {score}")
        return score

    def print_vifs(self):
        with open(f"../data/{self.name}_vifs.txt", "w") as text_file:
            for idx, col in enumerate(self.X.columns):
                print(f"{col}, {outliers_influence.variance_inflation_factor(self.X.values,idx)}",file=text_file)

    def print_coefs(self):
        with open(f"../data/{self.name}_coefs.txt", "w") as text_file:
            for idx, col in enumerate(self.columns):
                print(f"{col}, {self.model.coef_[idx]}",file=text_file)

    def print_goldsfeltquandt(self):
        linear_model = sm.OLS(self.y_train, self.X_train).fit()
        print(diagnostic.het_goldfeldquandt(linear_model.resid, linear_model.model.exog))

    def print_linear_summary(self):
        linear_model = sm.OLS(self.y_train, self.X_train).fit()
        with open(f"../data/{self.name}_linear_summary.txt", "w") as text_file:
            print(linear_model.summary2(),file=text_file)

    def try_imputes_scores(self):
        impute_scores = []
        for m in self.methods:
            self.clean_split_fit(m)
            impute_scores += [(m.__class__.__name__,self.print_score(),self.print_score('Test'))]
        with open(f"../data/{self.name}_impute_scores.txt", "w") as text_file:
            [print(f'{_[0]}, {_[1]}, {_[2]}',file=text_file) for _ in impute_scores]
        return impute_scores

if __name__ == '__main__':
    df = pd.read_csv('../data/PIRUS.csv',na_values=['-99'])
    PIRUS_Lasso = LinearModel(df, LassoCV(cv=10), 'Violent','Lasso')
    PIRUS_Elastic = LinearModel(df, ElasticNetCV(cv=10), 'Violent','ElasticNet')
    PIRUS_Lasso.clean_split_fit()
    PIRUS_Elastic.clean_split_fit()
    PIRUS_Lasso.print_score()
    PIRUS_Lasso.print_score(False)
    PIRUS_Elastic.print_score()
    PIRUS_Elastic.plot_mse()
    plt.close()
    PIRUS_Elastic.plot_coef_log_alphas()
    plt.close()
    PIRUS_Elastic.plot_scores_kfold()
    plt.close()
    PIRUS_Lasso.plot_mse()
    plt.close()
    PIRUS_Lasso.plot_coef_log_alphas()
    plt.close()
    PIRUS_Lasso.plot_scores_kfold()
    plt.close()
