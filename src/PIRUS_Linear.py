from init_data import Data
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

    def get_alphas(self):
        self.alphas = self.model.alphas_
        self.log_alphas = np.log10(self.alphas)

    def fit_model(self):
        self.model.fit(self.X_train,self.y_train)

    def plot_coef_log_alphas(self):
        coeffs = self.model.path(self.X_train,self.y_train)[1]
        plt.plot(self.log_alphas,coeffs.T)
        plt.axvline(np.log10(self.model.alpha_),linestyle='--')
        plt.title(f'Coefficient Descent of {self.name}')
        plt.xlabel(r'log($\alpha$)')
        plt.ylabel('Coefficients')
        plt.savefig(f'../plots/{self.name}_{self.predict}_coefficient_descent.png')

    def plot_mse(self):
        mse_path = self.model.mse_path_[:,1:]
        mean_mse = mse_path.mean(axis=1)
        plt.plot(self.log_alphas,mse_path,linestyle='--')
        plt.plot(self.log_alphas,mean_mse,label='Mean MSE',linewidth=5,color='k')
        # plt.title(r'MSE vs log($\alpha$)')
        plt.xlabel(r'log($\alpha$)')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(f'../plots/{self.name}_{self.predict}_MSE_plot.png')

    def plot_scores_kfold(self):
        lasso = Lasso(random_state=0)
        alphas = np.logspace(-4, -0.5, 30)

        tuned_parameters = [{'alpha': alphas}]
        n_folds = 5

        clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
        clf.fit(self.X_train, self.y_train)
        scores = clf.cv_results_['mean_test_score']
        scores_std = clf.cv_results_['std_test_score']
        plt.semilogx(alphas, scores)

        # plot error lines showing +/- std. errors of the scores
        std_error = scores_std / np.sqrt(n_folds)

        plt.semilogx(alphas, scores + std_error, 'b--')
        plt.semilogx(alphas, scores - std_error, 'b--')

        # alpha=0.2 controls the translucency of the fill color
        plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

        plt.ylabel('CV score +/- std error')
        plt.xlabel(r'$\alpha$')
        plt.axhline(np.max(scores), linestyle='--', color='.5')
        plt.xlim([alphas[0], alphas[-1]])
        plt.savefig(f'../plots/{self.name}_{self.predict}_kfold_mean_scores.png')

    def print_score(self):
        score = self.model.score(self.X_test,self.y_test)
        print(f"{self.name} Score (predict {self.predict}): {score}")

    def print_alpha(self):
        print(f"{self.name} Alpha (predict {self.predict}): {self.model.alpha_}")

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
        self.get_alphas()

if __name__ == '__main__':
    df = pd.read_csv('../data/PIRUS.csv',na_values=['-99'])
    PIRUS_Lasso = LinearModel(df, LassoCV(cv=10), 'Violent','Lasso')
    PIRUS_Elastic = LinearModel(df, ElasticNetCV(cv=10), 'Violent','ElasticNet')
    PIRUS_Lasso.clean_split_fit()
    PIRUS_Elastic.clean_split_fit()
    PIRUS_Lasso.print_score()
    PIRUS_Lasso.print_alpha()
    PIRUS_Elastic.print_score()
    PIRUS_Elastic.print_alpha()
    # PIRUS_Lasso.plot_mse()
    # plt.show()
    # PIRUS_Lasso.plot_coef_log_alphas()
    # plt.show()
    # PIRUS_Lasso.plot_scores_kfold()
    # plt.show()
