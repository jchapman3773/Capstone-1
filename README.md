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
1100 out of the 1865 (59%) of the individuals were part of a violent act.

Plot of the correlation heatmap for all the chosen features:
![heatmap](https://github.com/jchapman3773/Capstone-1/blob/master/plots/Correlation_Heatmap.png)
# Model
Feature Engineering
Modeling

# Results

# Future Work
Coef sometimes does not converge
