# Predicting Radicalized Individuals
Project for Data Science Immersive at Galvanize

by Julia Chapman

# Overview
__Data__: Individual Radicalization in the United States (PIRUS)

[__Source__](http://www.start.umd.edu/data-tools/profiles-individual-radicalization-united-states-pirus): National Consortium for the Study of Terrorism and Responses to Terrorism

This data covers backgrounds, attributes, and radicalization processes of over 1,800 violent and non-violent extremists who adhere to far right, far left, Islamist, or single issue ideologies in the United States from 1948-2016. These idividuals have radicalized within the United States to the point of committing ideologically motivated illegal violent or non-violent acts, 
joining a designated terrorist organization, or associated with an extremist organization whose leaders have been indicted of an ideologically motivated violent offense. 

My goal is to create a model that predicts, based on key features, whether the individual was part of a violent or non-violent act. The features that I selected are mostly demographic and socioeconomic data, with some additional features cherry picked from the dataset.

# Data
The dataset has 110 features (145 columns due to multiple columns for feature category). Unknown values were imputted as -99, so I imported -99 as NaN. -88 was used as an imput for "Does not apply". For example, in Language_English, -88 represented native english speakers, 1 for non-native english speakers, and 0 for non-english speakers. Because of this, I manually checked the codebook for each column with -88s and replaced them with NaN, 1, 0, etc based on my judgment.

The chosen columns were: | - | (descriptions available in the attached codebook)
--- | --- | ---
Broad_Ethnicity | Education | Group_Membership
Age | Student | Length_Group
Marital_Status | Employment_Status | Radical_Behaviors
Children, Gender | Military | Radical_Beliefs
Religious_Background | Foreign_Military | Abuse_Child
Convert | Social_Stratum_Childhood | Psychological
Reawakening | Aspirations | Alcohol_Drug
Residency_Status | Violent | Close_Family
Time_US_Months | Plot_Target1 | Previous_Criminal_Activity
Immigrant_Generation | Criminal_Severity | Angry_US
Language_English | Current_Status | -

## EDA
1100 out of the 1865 (59%) of the individuals were involved in a violent act.

Plot of the correlation heatmap for all the chosen features:
![heatmap](https://github.com/jchapman3773/Capstone-1/blob/master/plots/Correlation_Heatmap.png)
Based on the heatmap, none of the features had a high correlation with 'Violent'

Most of the Variance Inflation Factors (VIF) were below 10. The highest ones were Age and Time_US_Months. Here, 'Age' means the individual's age at time of exposure, when their activities/plot first came to public attention.

Feature | VIF | Feature | VIF
--- | --- | --- | ---
Broad_Ethnicity | 14.250 | Foreign_Military | 1.037
Age | 62.615| Social_Stratum_Childhood | 19.012
Marital_Status | 11.636 | Aspirations | 5.913
Children | 2.440 | Plot_Target1 | 2.516
Gender | 10.492 | Criminal_Severity | 13.023
Religious_Background | 9.836 | Current_Status | 5.297
Convert | 1.519 | Group_Membership | 5.381
Reawakening | 1.302 | Length_Group | 1.899
Residency_Status | 9.179 | Radical_Behaviors | 28.379
Time_US_Months | 46.984 | Radical_Beliefs | 9.198
Immigrant_Generation | 1.981 | Abuse_Child | 1.195
Language_English | 20.567 | Psychological | 1.362
Education | 10.5602 | Alcohol_Drug | 1.280
Student | 2.242 | Close_Family | 36.276
Employment_Status | 4.862 | Previous_Criminal_Activity | 2.133
Military | 1.305 | Angry_US | 6.608

![Age vs Time_US_Months](https://github.com/jchapman3773/Capstone-1/blob/master/plots/Time_US_Months_vs_Age.png)

# Model
Because my data had NaN values and was slightly imbalanced (~60%), I used fancyimpute's KNN and SMOTE.
I chose KNN 5 beacuse it looked to have a good balance of lower mse and higher model score. After imputing and balancing, I also scaled my data before splitting 25% for test/train.

Method | MSE | Model Train Score
--- | --- | ---
SimpleFill | 1775 | 0.632
KNN 1 | 1836 | 0.774
KNN 2 | 1800 | 0.788
KNN 3 | 1827 | 0.821
KNN 4 | 18354 | 0.829
KNN 5 | 1823 | 0.826
IterativeSVD | 74 | 0.62
MatrixFactorization | 1858 | 0.719

Before fitting my data, I used scikit-learn's RFE to select for 22 of my 32 features to make it easier to model.

At first, I tried LassoCV and ElasticNetCV as my models, but they didn't perform very well.

Lasso Mean Score:
![Lasso Mean Score](https://github.com/jchapman3773/Capstone-1/blob/master/plots/Lasso_Violent_kfold_mean_scores.png)

Lasso Mean MSE:
![Lasso Mean MSE](https://github.com/jchapman3773/Capstone-1/blob/master/plots/Lasso_Violent_MSE_plot.png)

Then I tried LogisticRegressionCV with L1 regularization as a penalizer. This performed much better. 
The test and train scores were similar every time the model was ran.

```
                          Results: Logit
==================================================================
Model:              Logit            Pseudo R-squared: 0.506      
Dependent Variable: y                AIC:              1173.0366  
Date:               2018-10-11 22:02 BIC:              1292.0242  
No. Observations:   1650             Log-Likelihood:   -564.52    
Df Model:           21               LL-Null:          -1143.7    
Df Residuals:       1628             LLR p-value:      4.7880e-232
Converged:          1.0000           Scale:            1.0000     
No. Iterations:     7.0000                                        
---------------------------------------------------------------------
        Coef.     Std.Err.       z       P>|z|      [0.025     0.975]
---------------------------------------------------------------------
x1     -0.2678      0.0916    -2.9246    0.0034    -0.4472    -0.0883  Broad_Ethnicity
x2      0.1439      0.2240     0.6424    0.5206    -0.2951     0.5829  Age
x3      0.1427      0.0924     1.5443    0.1225    -0.0384     0.3239  Marital_Status
x4     -0.1583      0.1055    -1.5002    0.1336    -0.3651     0.0485  Gender
x5      0.3151      0.0751     4.1971    0.0000     0.1680     0.4623  Religious_Background
x6     -0.9015      0.1066    -8.4547    0.0000    -1.1105    -0.6925  Convert
x7      0.5577      0.0959     5.8172    0.0000     0.3698     0.7457  Reawakening
x8      0.0784      0.0910     0.8615    0.3890    -0.1000     0.2568  Immigrant_Generation
x9      0.1498      0.2235     0.6704    0.5026    -0.2882     0.5878  Language_English
x10     0.7189      0.1252     5.7423    0.0000     0.4736     0.9643  Student
x11    -0.1558      0.0838    -1.8593    0.0630    -0.3199     0.0084  Social_Stratum_Childhood
x12    -0.2476      0.0978    -2.5314    0.0114    -0.4394    -0.0559  Aspirations
x13     0.4277      0.0987     4.3313    0.0000     0.2342     0.6212  Plot_Target1
x14     1.2913      0.1313     9.8323    0.0000     1.0339     1.5487  Criminal_Severity
x15     0.1095      0.0885     1.2377    0.2158    -0.0639     0.2829  Current_Status
x16    -0.1195      0.0911    -1.3117    0.1896    -0.2980     0.0590  Length_Group
x17     1.4687      0.1444    10.1709    0.0000     1.1857     1.7517  Radical_Behaviors
x18    -0.1105      0.0795    -1.3891    0.1648    -0.2664     0.0454  Abuse_Child
x19     0.1725      0.0841     2.0515    0.0402     0.0077     0.3373  Psychological
x20     0.0969      0.0817     1.1862    0.2355    -0.0632     0.2571  Alcohol_Drug
x21     0.0917      0.0837     1.0955    0.2733    -0.0724     0.2559  Close_Family
x22     0.1664      0.0834     1.9952    0.0460     0.0029     0.3299  Angry_US
==================================================================
```

Logistic Regression Coefficient Descent:
![Coefficient Descent](https://github.com/jchapman3773/Capstone-1/blob/master/plots/LogisticRegression_Violent_coefficient_descent.png)

Logistic Regression Mean Score:
![Mean Scores](https://github.com/jchapman3773/Capstone-1/blob/master/plots/LogisticRegression_Violent_kfold_mean_scores.png)

Logistic Regression ROC Curve:
![ROC Curve](https://github.com/jchapman3773/Capstone-1/blob/master/plots/LogisticRegression_Violent_ROC_curve.png)

# Results
Confusion Matrix:

-- | P | N
-- | -- | --
P | 223 | 47
N | 42 | 238

```
              precision    recall  f1-score   support

         0.0       0.84      0.83      0.83       270
         1.0       0.84      0.85      0.84       280
```
(precision = tp / (total_predict_P))
(recall = tp / (total_true_P)

# Future Work
When running my final model, it would often not converge. I made it happen less often by increasing the max_iter from 100 to 500, but it would still occur. In the future, I would like to figure out why the model isn't converging and make it better.
