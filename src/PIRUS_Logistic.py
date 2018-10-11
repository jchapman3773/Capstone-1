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
from sklearn.metrics import roc_curve,auc
from statsmodels.stats import outliers_influence, diagnostic
from fancyimpute import SimpleFill, KNN, IterativeSVD, MatrixFactorization
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

    def clean_split_fit(self,impute=KNN(5)):
        self.prep_data(impute)
        self.fit_model()
        self.get_Cs()

    def plot_coef_log_alphas(self):
        plt.plot(self.log_Cs, self.model.coefs_paths_[1][1])
        plt.axvline(np.log10(self.model.C_),linestyle='--')
        plt.title(f'Coefficient Descent of {self.name}')
        plt.xlabel('log(C)')
        plt.ylabel('Coefficients')
        line_names = np.append(['Intercept'],self.X.columns.values)
        plt.legend(np.append(line_names,['Chosen C']), fontsize = 'x-small',loc='upper right')
        plt.savefig(f'../plots/{self.name}_{self.predict}_coefficient_descent.png')

    def plot_scores_kfold(self):
        logistic = LogisticRegression(penalty='l1', solver='saga', random_state=0)
        Cs = np.logspace(-4, 4, 30)

        tuned_parameters = [{'C': Cs }]

        clf = GridSearchCV(logistic, tuned_parameters, cv=10, refit=False)
        clf.fit(self.X_train, self.y_train)
        scores = clf.cv_results_['mean_test_score']
        scores_std = clf.cv_results_['std_test_score']
        plt.semilogx(Cs, scores)

        std_error = scores_std / np.sqrt(10)
        plt.semilogx(Cs, scores + std_error, 'b--')
        plt.semilogx(Cs, scores - std_error, 'b--')
        plt.fill_between(Cs, scores + std_error, scores - std_error, alpha=0.2)

        plt.title('CV score +/- std error')
        plt.ylabel('Score')
        plt.xlabel('Cs')
        plt.axhline(np.max(scores), linestyle='--', color='.5')
        plt.savefig(f'../plots/{self.name}_{self.predict}_kfold_mean_scores.png')

    def plot_ROC(self):
        y_prob = self.model.predict_proba(self.X_test)[:,1]
        fpr, tpr, _ = roc_curve(self.y_test,y_prob)
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC: {round(roc_auc,2)})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(f'../plots/{self.name}_{self.predict}_ROC_curve')

    def print_score(self,test_train='Train'):
        if test_train == 'Test':
            score = self.model.score(self.X_test,self.y_test)
        else:
            score = self.model.score(self.X_train,self.y_train)
        print(f"{test_train} Score: {score}")
        return score

    def print_vifs(self):
        for idx, col in enumerate(self.X_train.columns):
            print(f"{col}: {outliers_influence.variance_inflation_factor(self.X_train.values,idx)}")

    def print_coefs(self):
        for idx, col in enumerate(self.X_train.columns):
            print(f"{col}: {self.model.coef_[idx]}")

    def print_logistic_summary(self):
        logistic_model = sm.Logit(self.y_train, self.X_train).fit()
        print(logistic_model.summary2())

    def try_imputes_scores(self):
        methods = [SimpleFill(), KNN(1), KNN(2), KNN(3), KNN(4), KNN(5), IterativeSVD(), MatrixFactorization()]
        impute_scores = []
        for m in methods:
            self.clean_split_fit(m)
            impute_scores += [(m.__class__.__name__,self.print_score(),self.print_score('Test'))]
        with open(f"../data/{self.name}_impute_scores.txt", "w") as text_file:
            [print(f'{_[0]}, {_[1]}, {_[2]}',file=text_file) for _ in impute_scores]
        return impute_scores

if __name__ == '__main__':
    df = pd.read_csv('../data/PIRUS.csv',na_values=['-99'])
    PIRUS = LogisticModel(df, LogisticRegressionCV(penalty='l1',cv=10,solver='saga'), 'Violent','LogisticRegression')
    PIRUS.clean_split_fit()
    # PIRUS.print_score()
    # PIRUS.print_score('Test')
    PIRUS.plot_coef_log_alphas()
    plt.close()
    PIRUS.plot_scores_kfold()
    plt.close()
    PIRUS.plot_ROC()
