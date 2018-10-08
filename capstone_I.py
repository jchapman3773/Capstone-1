# from scipy import stats
# import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    'figure.figsize'      : (10,10),
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'large',
    'legend.loc'          : 'upper right'
})

df = pd.read_csv('PIRUS.csv')

df[['Loc_Plot_State1','Plot_Target1','Date_Exposure','Attack_Preparation','Violent','Criminal_Severity','Gang_Age_Joined']].hist(bins=10)
plt.show()
