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
from sklearn.svm import l1_min_c
from sklearn.metrics import roc_curve,auc,confusion_matrix,classification_report
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
        self.selector = None
        self.columns = None
        self.scaler = XyScaler()

    def select_features(self,features=22):
        selector = RFE(self.model,features)
        self.selector = selector.fit(self.X_train,self.y_train)
        self.X_train = self.X_train[:,self.selector.support_]
        self.X_test = self.X_test[:,self.selector.support_]
        self.columns = self.X.loc[:,self.selector.support_].columns

    def fit_model(self):
        self.model.fit(self.X_train,self.y_train)

    def get_Cs(self):
        self.Cs = self.model.Cs_
        self.log_Cs = np.log10(self.Cs)

    def clean_split_fit(self,impute=KNN(5)):
        self.prep_data(impute)
        self.select_features()
        self.fit_model()
        self.get_Cs()

    def plot_coef_log_alphas(self):
        plt.plot(self.log_Cs, self.model.coefs_paths_[1][1])
        plt.axvline(np.log10(self.model.C_),linestyle='--')
        plt.title(f'Coefficient Descent of {self.name}')
        plt.xlabel('log(C)')
        plt.ylabel('Coefficients')
        line_names = np.append(['Intercept'],self.columns.values)
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

        plt.title(f'{self.name} CV score +/- std error')
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
        plt.title(f'{self.name} ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(f'../plots/{self.name}_{self.predict}_ROC_curve')

    def print_score(self,test=True):
        if test:
            score = self.model.score(self.X_test,self.y_test)
        else:
            score = self.model.score(self.X_train,self.y_train)
        print(f"{'Test' if test else 'Train'} Score: {score}")
        return score

    def print_confusion_matrix(self):
        tn, fp, fn, tp = confusion_matrix(self.y_test,self.model.predict(self.X_test)).ravel()
        with open(f"../data/{self.name}_confusion_matrix.txt", "w") as text_file:
            print(f'{tn} | {fp}\n{fn} | {tp}',file=text_file)
        return tn, fp, fn, tp

    def print_classification_report(self):
        with open(f"../data/{self.name}_classification_report.txt", "w") as text_file:
            print(classification_report(self.y_test,self.model.predict(self.X_test)),file=text_file)

    def print_vifs(self):
        with open(f"../data/{self.name}_vifs.txt", "w") as text_file:
            for idx, col in enumerate(self.X.columns):
                print(f"{col} | {outliers_influence.variance_inflation_factor(self.X.values,idx)}",file=text_file)

    def print_coefs(self):
        with open(f"../data/{self.name}_coefs.txt", "w") as text_file:
            for idx, col in enumerate(self.columns):
                print(f"{col} | {self.model.coef_[0][idx]}",file=text_file)

    def print_logistic_summary(self):
        logistic_model = sm.Logit(self.y_train, self.X_train).fit()
        with open(f"../data/{self.name}_logistic_summary.txt", "w") as text_file:
            print(logistic_model.summary2(),file=text_file)

    def try_imputes_scores(self):
        impute_scores = []
        for m in self.methods:
            self.clean_split_fit(m)
            impute_scores += [(m.__class__.__name__,self.print_score(),self.print_score(True))]
        with open(f"../data/{self.name}_impute_scores.txt", "w") as text_file:
            [print(f'{_[0]} | {_[1]} | {_[2]}',file=text_file) for _ in impute_scores]
        return impute_scores

if __name__ == '__main__':
    df = pd.read_csv('../data/PIRUS.csv',na_values=['-99'])
    PIRUS = LogisticModel(df, LogisticRegressionCV(penalty='l1',cv=10,solver='saga',max_iter=500), 'Violent','L1_LogisticRegression')
    PIRUS.clean_split_fit()

    PIRUS.print_score()
    PIRUS.print_score(False)

    PIRUS.plot_coef_log_alphas()
    plt.close()
    PIRUS.plot_scores_kfold()
    plt.close()
    PIRUS.plot_ROC()
    plt.close()
    # PIRUS.make_heatmap()

    PIRUS.print_coefs()
    PIRUS.print_vifs()
    PIRUS.print_logistic_summary()
    PIRUS.print_confusion_matrix()
    PIRUS.print_classification_report()

    [print(col) for col in PIRUS.columns]
