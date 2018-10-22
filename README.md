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

The chosen columns were: |   | (descriptions available in the attached codebook)
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
Language_English | Current_Status |  

## EDA
1100 out of the 1865 (59%) of the individuals were involved in a violent act.

Plot of the correlation heatmap for all the chosen features:
![heatmap](https://github.com/jchapman3773/Capstone-1/blob/master/plots/Correlation_Heatmap.png)
Based on the heatmap, none of the features had a high correlation with 'Violent'

Most of the Variance Inflation Factors (VIF) were below 10. The highest ones were Age and Time_US_Months.

Here, 'Age' means the individual's age at time of exposure, when their activities/plot first came to public attention.

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
KNN 4 | 1835 | 0.829
KNN 5 | 1823 | 0.826
IterativeSVD | 74 | 0.62
MatrixFactorization | 1858 | 0.719

Before fitting my data, I used scikit-learn's RFE to select for 22 of my 32 features to make it easier to model.

For my model, I used LogisticRegressionCV with L1 regularization as a penalizer.

The test and train scores were similar every time the model was ran.

Gender had the most negative coefficient, but it might be biased because only about 10% of the data's population were female.

Abuse_Child and Criminal_Severity both had high positive coefficients. Criminal_Severity is the extent of the individual's criminal activity by time of exposure, and so probably has intisic correlation to the individual's participation in violent acts.

```
                          Results: Logit
==================================================================
Model:              Logit            Pseudo R-squared: 0.511      
Dependent Variable: y                AIC:              1162.2929  
Date:               2018-10-12 08:50 BIC:              1281.2805  
No. Observations:   1650             Log-Likelihood:   -559.15    
Df Model:           21               LL-Null:          -1143.7    
Df Residuals:       1628             LLR p-value:      2.4278e-234
Converged:          1.0000           Scale:            1.0000     
No. Iterations:     7.0000                                        
---------------------------------------------------------------------
        Coef.     Std.Err.       z       P>|z|      [0.025     0.975]
---------------------------------------------------------------------
x1     -0.3831      0.0900    -4.2567    0.0000    -0.5595    -0.2067  Broad_Ethnicity
x2      0.4008      0.1116     3.5913    0.0003     0.1821     0.6196  Age
x3     -0.1161      0.0981    -1.1832    0.2367    -0.3084     0.0762  Marital_Status
x4      0.2836      0.0748     3.7927    0.0001     0.1370     0.4301  Children
x5     -0.8680      0.1064    -8.1603    0.0000    -1.0764    -0.6595  Gender
x6      0.6769      0.0981     6.8967    0.0000     0.4845     0.8692  Religious_Background
x7      0.0989      0.0980     1.0088    0.3131    -0.0933     0.2911  Convert
x8      0.7194      0.1084     6.6390    0.0000     0.5070     0.9318  Reawakening
x9      0.0970      0.0972     0.9982    0.3182    -0.0935     0.2875  Time_US_Months
x10    -0.0791      0.0968    -0.8169    0.4140    -0.2689     0.1107  Immigrant_Generation
x11    -0.1159      0.0826    -1.4024    0.1608    -0.2779     0.0461  Employment_Status
x12    -0.2310      0.0958    -2.4111    0.0159    -0.4187    -0.0432  Aspirations
x13     0.4488      0.0999     4.4950    0.0000     0.2531     0.6446  Plot_Target1
x14     1.4089      0.1334    10.5647    0.0000     1.1475     1.6703  Criminal_Severity
x15     0.0664      0.0883     0.7519    0.4521    -0.1067     0.2395  Current_Status
x16     0.2008      0.0876     2.2912    0.0220     0.0290     0.3726  Length_Group
x17    -0.1417      0.1066    -1.3290    0.1839    -0.3506     0.0673  Radical_Behaviors
x18     1.2652      0.1340     9.4429    0.0000     1.0026     1.5278  Abuse_Child
x19     0.1230      0.0775     1.5862    0.1127    -0.0290     0.2750  Psychological
x20     0.1578      0.0815     1.9360    0.0529    -0.0020     0.3175  Close_Family
x21     0.1446      0.0847     1.7081    0.0876    -0.0213     0.3106  Previous_Criminal_Activity
x22     0.1615      0.0824     1.9590    0.0501    -0.0001     0.3231  Angry_US
==================================================================

```

![Coefficient Descent, L1](https://github.com/jchapman3773/Capstone-1/blob/master/plots/L1_LogisticRegression_Violent_coefficient_descent.png)

![Mean Scores, L1](https://github.com/jchapman3773/Capstone-1/blob/master/plots/L1_LogisticRegression_Violent_kfold_mean_scores.png)

![ROC Curve, L1](https://github.com/jchapman3773/Capstone-1/blob/master/plots/L1_LogisticRegression_Violent_ROC_curve.png)

Logistic regression with a L2 penalizer had different coefficient paths, but the end results were the same.

![Coefficient Descent, L2](https://github.com/jchapman3773/Capstone-1/blob/master/plots/L2_LogisticRegression_Violent_coefficient_descent.png)


# Results
Confusion Matrix:

-- | P | N
-- | -- | --
P | 221 | 49
N | 47 | 233

Classification Report:
```
              precision    recall  f1-score   support

         0.0       0.82      0.82      0.82       270
         1.0       0.83      0.83      0.83       280

         avg       0.83      0.83      0.83       550
```

Of the predicted Violent acts, 83% were correctly identified.

My model was able to correctly identify 83% of the Violent acts.

# Future Work
When running my final model, it would often not converge. I made it happen less often by increasing the max_iter from 100 to 500, but it would still occur. In the future, I would like to figure out why the model isn't converging and make it better.

Future Analysis would be to put in all the feature from the dataset and select 20 or so using RFE and see if it improves the model. SOme features that have sparse data could also be removed by looking at missingno. Age or TIme_US_Months should be removed for multicollinearity. Criminal_Severity should be removed since it is intrinsically correlated.

Script for the linear LassoCV and ElasticNetCV models could also be used to predict continuous features like Age.

Census data could be brought in to determine if the population in the PIRUS data set are statistically different from the average population.
