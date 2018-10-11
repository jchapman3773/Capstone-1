from init_data import Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import XyScaler
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV, LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import l1_min_c
from sklearn.metrics import roc_curve
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

class LogisticModel(Data):

    def __init__(self,data,model,predict,name):
        super().__init__(data,predict)
        self.model = model
        self.name = name
        self.Cs = None
        self.log_Cs = None
        self.scaler = XyScaler()

    def fit_model(self):
        self.model.fit(self.X_train,self.y_train)

    def get_Cs(self):
        self.Cs = self.model.Cs_
        self.log_Cs = np.log10(self.Cs)

    def clean_split_fit(self):
        self.prep_data()
        self.fit_model()
        self.get_Cs()

    def plot_coef_log_alphas(self):
        plt.plot(self.log_Cs, self.model.coefs_paths_[1][1])
        # coeffs = self.model.coefs_paths_.values
        plt.axvline(np.log10(self.model.C_),linestyle='--')
        plt.title(f'{self.name} Path')
        plt.xlabel('log(C)')
        plt.ylabel('Coefficients')
        plt.savefig(f'../plots/{self.name}_{self.predict}_coefficient_descent.png')

    def plot_mse(self):
        mse_path = self.model.mse_path_
        mean_mse = mse_path.mean(axis=1)
        plt.plot(self.log_alphas,mse_path,linestyle='--')
        plt.plot(self.log_alphas,mean_mse,label='Mean MSE',linewidth=5,color='k')
        plt.title(r'MSE vs log($\alpha$)')
        plt.xlabel(r'log($\alpha$)')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(f'../plots/{self.name}_{self.predict}_MSE_plot.png')

    def plot_scores_kfold(self):
        logistic = LogisticRegression(penalty='l1', solver='saga', random_state=0)
        Cs = np.logspace(-4, 4, 30)

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
        plt.savefig(f'../plots/{self.name}_{self.predict}_kfold_mean_scores.png')

    def plot_ROC(self):
        # fpr,tpr = roc_curve(self.y_train,self.model.scores_[1])
        y_prob = self.model.predict_proba(self.X_train)[:,1]
        import pdb; pdb.set_trace()
        fpr, tpr, _ = roc_curve(self.y_train,y_prob)
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig(f'../plots/{self.name}_{self.predict}_ROC_curve')

    def print_score(self,test_train='Train'):
        if test_train == 'Test':
            score = self.model.score(self.X_test,self.y_test)
        else:
            score = self.model.score(self.X_train,self.y_train)
        print(f"{test_train} Score: {score}")

    def print_vifs(self):
        for idx, col in enumerate(self.X_train.columns):
            print(f"{col}: {outliers_influence.variance_inflation_factor(self.X_train.values,idx)}")

    def print_coefs(self):
        for idx, col in enumerate(self.X_train.columns):
            print(f"{col}: {self.model.coef_[idx]}")

    def print_linear_summary(self):
        logistic_model = sm.Logit(self.y_train, self.X_train).fit()
        print(logistic_model.summary2())

if __name__ == '__main__':
    df = pd.read_csv('../data/PIRUS.csv',na_values=['-99'])
    PIRUS = LogisticModel(df, LogisticRegressionCV(penalty='l1',cv=10,solver='saga'), 'Violent','LogisticRegression')
    PIRUS.clean_split_fit()
    # PIRUS.try_many_imputes()
    PIRUS.print_score()
    PIRUS.print_score('Test')
    # PIRUS.plot_coef_log_alphas()
    # plt.show()
    # PIRUS.plot_scores_kfold()
    # plt.show()
    # PIRUS.plot_mse()
    # plt.show()
    PIRUS.plot_ROC()
