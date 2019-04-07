# -*- coding: utf-8 -*-
"""
Aim: To utilize the machine learning framework to develop a model that 
best analyszes and predicts an infant's birth weight based on various 
characteristics.

Contents : 
    1. Importing Libraries
    2. Exploratory Data Analysis (EDA)
        - Dataset exploration
        - Missing values
        - Grouping of variables
        - Outliers , boxplots
        - Histograms
        - Scatter Plots
        - Correleations
    3. OLS 
    4. Train & Test Predictions
       - Linear  
       - KNN 
       - Ridge
       - Lasso 
    6. Compare models
          
"""
###############################################################################
# Importing libraries
###############################################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf # regression modeling

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso



file = 'birthweight1.xlsx'
birth = pd.read_excel(file)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

###############################################################################
# Fundamental Dataset Exploration
###############################################################################

# Column names
birth.columns

# Dimensions of the DataFrame
birth.shape

# Information about each variable
birth.info()


# Descriptive statistics
desc = birth.describe().round(2)

print(desc)

###############################################################################
# Imputing Missing Values
###############################################################################


print(
      birth
      .isnull()
      .sum()
      )

print(
      ((birth[:].isnull().sum())
      /
      birth[:]
      .count())
      .round(2)
      )
        
"""
Missing values are in meduc, npvis, feduc

# Check if missing values follow a normal distribution / skewed

"""

df_dropped = birth.dropna()


sns.distplot(df_dropped['meduc'])
sns.distplot(df_dropped['npvis'])
sns.distplot(df_dropped['feduc'])

# Check for main variable bwght

sns.distplot(birth['bwght'])

# Flagging missing values

for col in birth:
    
    #print(col)
    
    """Create columns that are 0s if a value was not missing and 1 if
    a value is missing."""
    
    if birth[col].isnull().astype(int).sum() > 0:
        
        birth['m_'+col] = birth[col].isnull().astype(int)


# Fill missing values with median

for col in birth:

    """ Impute median values using the mean of each column """

    if birth[col].isnull().astype(int).sum() > 0:

        col_median = birth[col].median()

        birth[col] = birth[col].fillna(col_median).round(2)

# Checking the overall dataset to see if there are any remaining
# missing values

print(
      birth
      .isnull()
      .any()
      .any()
      )


###############################################################################
# Grouping of Variables (Feature Engineering)
###############################################################################

"""
Rationales for all the Feature Engineering Variables are provided below. 

Some of the feature engineering variables were created but not used in the
final model due to its signficance being low. However, the critical variables 
were compared outside the model and further analysis can be found below in 
script. Refer to Lines 940 onwards. 

"""


"""
From research (https://kidshealth.org/en/parents/grownewborn.html), 
healthy babies weigh in the range of 2500 grams and 4000 grams.
Any baby below 2500 grams is classfied as underweight and above 4000 grams is
classified as overweight. Mostly, reasons for underweight babies are when they
are born pre-maturely (Research: https://www.urmc.rochester.edu/encyclopedia/
content.aspx?contenttypeid=90&contentid=p02382) or there is some issue with 
them. In this dataset, no information is provided on whether the baby was born 
pre-mature or not, hence the underweight babies are grouped together. 

From the value counts, in our dataset, there are 15 babies (8% of total data)
are underweight. This is a small sample size however, some characteristics are
observed below to try and form prelimiary hypothesis/understanding of the 
reasons for low birth weight. 

"""


def func(x):
    
    if x < 2500:
        return 1
    else:
        return 0

birth['new_bwght'] = birth['bwght'].map(func)

birth['new_bwght'].value_counts()


def func(x):
    
    if 2500 <= x <= 4000:
        return 1
    else:
        return 0

birth['new_bwght_2'] = birth['bwght'].map(func)


"""
From this data's outlier analysis (below in python script), it was identified:

# High Baby Weight (>4000 grams), n= 18 babies, avg(mother's age) = 36

# Low Baby Weight (<2500 grams), n= 15 babies, avg(mother's age) = 54

This implies that focusing on our analysis of low baby's weight, the older the 
mother, she seems to have an influence / tendency of giving birth to a baby
with low birth weight as opposed to a baby with high birth weight. 

Hence, for the model to predict low birth weight, the mother's age threshold 
was set to 55. Different thresholds were also tried, it did not really make a 
difference whether it was 54 or 56, hence a round number was selected. 

From research (https://www.yourfertility.org.au/everyone/age), 
a women's fertility reduces after age 35. However from our dataset, if we were
to set the threshold at 35, we do not get any meaningful analysis generated 
as the model prediction is poor when we use mother's age as one of the 
variables. 

"""

def func(x):
    
    if x > 55:
        return 1
    else:
        return 0

birth['new_mage'] = birth['mage'].map(func)

"""
From research (https://www.webmd.com/baby/guide/prenatal-vitamins#2),
it is recommended that mothers should start taking their pre-natal vitamins
between zero and three months which is focul for baby's wellbeing.

"""

def func(x):
    if 0 <= x <= 3:
        return 0
    else:
        return 1

birth['new_monpre_2'] = birth['monpre'].map(func)

"""
From research (https://www.webmd.com/baby/how-often-do-i-need-prenatal-visits),
there is a recommended number of pre-natal visits based on the month of 
pregnancy. Using a back-calcualtion of converting weeks to months, we
decided on the threshold of 12 for the number of pre-natal visits. 

"""

def func(x):
    if x >= 12:
        return 1
    else:
        return 0

birth['new_npvis_2'] = birth['npvis'].map(func)



"""
Research (https://www.sciencedirect.com/science/article/pii/S0021755713000971)
indicates that mothers who are highly educated have 33% less chance of having
a baby with low birth weight as compared to mothers who are medium educated.
This is because the education level influences the social economical status of
the mother, and how well she can take care of herself during pregnancy. 

For education, a threshold of 12 years for the mother's and father's age 
individually was selected. This is because an education beyond 12 years implied 
the mother and/or father attended university. Generally, if someone has
attained a university education, that person is categorised as well-educated
(https://www.wg.aegee.org/ewg/higheredu.htm). 

For our dataset, the average combined education of the mother and father 
was also calculated as sometimes having both parents educated is more important
than only have one as this is impacted by which parent is more vocal/dominant/
aggressive.

"""

def func(x):
    if x > 12:
        return 1
    else:
        return 0

birth['new_meduc'] = birth['meduc'].map(func)


def func(x):
    if x > 12:
        return 1
    else:
        return 0

birth['new_feduc'] = birth['feduc'].map(func)


birth['avg_combined_educ'] = (birth['meduc'] + birth['feduc'])/2

"""
From research (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4199306/), it seems
that there can be an influence on the baby's birth weight if the parents are of
the same race or not. For instance, the research said for African-American 
women, there is a strong chance that they will have babies with low birth 
weight as opposed to other races.  
Hence, the following race groups were engineered. 
 
"""


# race_group_1


def fn1(row):
    
    if (row['mwhte'] and row['fwhte']):
        
        return 4
    
    elif (row['mwhte'] and row['fblck']):
        
        return 3

    elif row['mblck'] and row['fwhte']:

        return 3

    elif row['mblck'] and row['fblck']:

        return 2

    elif row['moth'] and row['foth']:

        return 0

    else:
        
        return 1


birth['new_race'] = birth.apply(fn1, axis = 1)


#### race_group_2


def fn1(row):
    if row['mwhte'] and row['fwhte']:
        
        return 1

    elif row['mblck'] and row['fblck']:

        return 1

    elif row['moth'] and row['foth']:

        return 1

    else:

        return 0


birth['new_race_2'] = birth.apply(fn1, axis=1)


"""
Research (https://americanpregnancy.org/is-it-safe/wine-during-pregnancy/) 
found indicates that there is concrete number of drinks that are considered
acceptable for mothers to drink when pregnant. Hence, for our analysis, it is 
considered as mothers who drink (>0) and who do not drink (0). 

Similar research (https://women.smokefree.gov/pregnancy-motherhood/quitting-
while-pregnant/myths-about-smoking-pregnancy) was found for cigarettes as well.
There is no concrete number of cigarettes allowed to smoke for mothers who 
are pregnant. 

"""



###############################################################################
# Outlier Analysis
###############################################################################


#Loop for flagging outliers

# numerical vars start from this index (without binary columns)

x=9

# this df only has the numerical columns

birth1 = pd.concat([birth.iloc[:,:x], birth['bwght']], axis = 1 )


#print(regions_team_umi)
# number of numerical variables

total_numvarumi = birth1.shape[1]

# calculate quantiles for numerical variables

iqr = 1.5*(birth1.quantile(0.75) - birth1.quantile(0.25))
birth_h = birth1.quantile(0.75) + iqr
birth_l = birth1.quantile(0.25) - iqr

#create outlier df

out_cols=pd.DataFrame()

for i in range(total_numvarumi):

	out_cols['outlier_'+birth1.keys()[i]] = ((birth1.iloc[:,i] > birth_h[i]) 
    | (birth1.iloc[:,i] < birth_l[i]))

# merge original df and outlier columns

birth = pd.concat([birth, out_cols], axis = 1)


### Further subsetting

# Subset the dataset for Low Baby Weight (smaller than 2,500g) and also 
# bigger than 4,000g

Higher = birth['bwght'] > 4000

Lower = birth['bwght'] < 2500

# High Baby Weight n= 18 babies, avg(mage)36, avg(cigs)~4.5, avg(drink) 2

higher_birth = birth[Higher]

desc_higher = higher_birth.describe().round(2)

print(desc_higher)

# Low Baby Weight n= 15 babies, avg(mage)54, avg(cigs)~17.5, avg(drink) 10.5

lower_birth = birth[Lower]

desc_lower = lower_birth.describe().round(2)

print(desc_lower)

# Take out mother's age outliers

no_old_moms = birth['outlier_mage'] == 0

birth_no_old_moms = birth[no_old_moms]

desc_bnom = birth_no_old_moms.describe().round(2) # describe

print(desc_bnom)

# Analyse mother's age outliers

old_moms = birth['outlier_mage'] == 1

birth_old_moms = birth[old_moms]

desc_bom = birth_old_moms.describe().round(2) # describe

print(desc_bom)

# The 5 of the 6 outliers in mother's age have LWB !!!
# Also the difference in the age of mothers from LWB to HWB is almost 
# 20 years


### Analyse npvis's outliers


# Prenatal visits Low limit Outliers

low_npvis = birth['npvis'] < 7

birth_low_npvis = birth[low_npvis]

desc_bln = birth_low_npvis.describe().round(2) # describe

print(desc_bln)

# Prenatal visits High limit Outliers 

high_npvis = birth['npvis'] > 15

birth_high_npvis = birth[high_npvis]

desc_bhn = birth_high_npvis.describe().round(2) # describe

print(desc_bhn)

# A third of the low outlier for npvis are also outliers to monp
# On average, the difference between high and low npvis in both parents
# education is ~ 2


#####PLOTS (According to IQR) #####

sns.boxplot(
            y = 'bwght',
            data = birth,
            width = 0.5,
            palette = 'colorblind'
            ) # Below 1800

sns.boxplot(
            y = 'mage',
            data = birth,
            width = 0.5,
            palette = 'colorblind'
            )  # Above 66  

sns.boxplot(
            y = 'meduc',
            data = birth,
            width = 0.5,
            palette = 'colorblind'
            ) # No outliers  

sns.boxplot(
            y = 'monpre',
            data = birth,
            width = 0.5,
            palette = 'colorblind'
            )# Above 4

sns.boxplot(
            y = 'npvis',
            data = birth,
            width = 0.5,
            palette = 'colorblind'
            ) # Below 7 and Above 15

sns.boxplot(
            y = 'feduc',
            data = birth,
            width = 0.5,
            palette = 'colorblind'
            ) # We have 2 at 1  

sns.boxplot(   
            y = 'cigs',
            data = birth,
            width = 0.5,
            palette = 'colorblind'
            )  # No outliers 

sns.boxplot(
            y = 'drink',
            data = birth,
            width = 0.5,
            palette = 'colorblind'
            ) # Above 12


# For birthweight by each column grouped


bp = sns.boxplot(
                 y = 'bwght',
                 x = 'new_mage',
                 data = birth,
                 width = 0.5,
                 palette = 'colorblind'
                 )

bp = sns.boxplot(
                 y = 'bwght',
                 x = 'male',
                 data = birth,
                 width = 0.5,
                 palette = 'colorblind'
                 )



###############################################################################
# Histograms
###############################################################################


plt.subplot(2, 2, 1)
sns.distplot(
             birth['omaps'],
             bins = 'fd',
             color = 'g'
             )

plt.xlabel('omaps')


plt.subplot(2, 2, 2)
sns.distplot(
             birth['fmaps'],
             bins = 'fd',
             color = 'y'
             )

plt.xlabel('fmaps')


plt.subplot(2, 2, 3)
sns.distplot(
             birth['feduc'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange'
             )

plt.xlabel('feduc')


plt.subplot(2, 2, 4)

sns.distplot(   
             birth['mblck'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r'
             )


plt.xlabel('mblck')

plt.tight_layout()
plt.savefig('birth Data Histograms.png')

########################

plt.show()
plt.subplot(2, 2, 1)

sns.distplot(
             birth['fblck'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r'
             )

plt.xlabel('fblck')


plt.subplot(2, 2, 2)

sns.distplot(
             birth['male'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r'
             )


plt.xlabel('male')
plt.subplot(2, 2, 3)

sns.distplot(
             birth['meduc'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r'
             )


plt.xlabel('meduc')
plt.subplot(2, 2, 4)

sns.distplot(   
             birth['npvis'],
             bins = 'fd',
             kde = False,
             rug = True,
             color = 'r'
             )

plt.xlabel('npvis')
plt.tight_layout()

plt.savefig('birth Data Histograms1.png')

########################

plt.show()
plt.subplot(2, 2, 1)

sns.distplot(
             birth['moth'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r'
             )

plt.xlabel('moth')

plt.subplot(2, 2, 2)

sns.distplot(
             birth['fwhte'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r'
             )

plt.xlabel('fwhte')
plt.subplot(2, 2, 3)

sns.distplot(
             birth['monpre'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r'
             )


plt.xlabel('monpre')
plt.subplot(2, 2, 4)

sns.distplot(
             birth['foth'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r'
             )

plt.xlabel('foth')
plt.tight_layout()

plt.savefig('birth Data Histograms2.png')

########################

plt.show()
plt.subplot(2, 2, 1)

sns.distplot(
             birth['mwhte'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r'
             )

plt.xlabel('mwhte')
plt.subplot(2, 2, 2)

sns.distplot(
             birth['fage'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r'
             )

plt.xlabel('fage')

plt.subplot(2, 2, 3)

sns.distplot(
             birth['mage'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r'
             )


plt.xlabel('mage')
plt.subplot(2, 2, 4)

sns.distplot(
             birth['cigs'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r'
             )


plt.xlabel('cigs')
plt.tight_layout()

plt.savefig('birth Data Histograms3.png')

########################
plt.show()

sns.distplot(
             birth['drink'],
             bins = 'fd',
             kde = False,
             rug = True,
             color = 'r'
             )


plt.xlabel('drink')
plt.tight_layout()

plt.savefig('birth Data Histograms4.png')


###############################################################################
# Scatter Plots
###############################################################################


# mother age & birthweight           ---- scatter! 
plt.scatter(
            x = 'mage',
            y = 'bwght',
            alpha = 0.2,
            color = 'blue',
            data = birth
            )

plt.title('mage & bwght')
plt.ylabel('bwght')
plt.grid(True)


###mother education and birthweight
plt.scatter(
            x = 'meduc',
            y = 'bwght',
            alpha = 0.2,
            color = 'blue',
            data = birth
            )

plt.title('meduc & bwght')
plt.ylabel('bwght')
plt.grid(True)


####monpre and birthweight
plt.scatter(
            x = 'monpre',
            y = 'bwght',
            alpha = 0.2,
            color = 'blue',
            data = birth
            )

plt.title('monpre & bwght')
plt.ylabel('bwght')
plt.grid(True)


###npvis and birthweight               ---- scatter!
plt.scatter(
            x = 'npvis',
            y = 'bwght',
            alpha = 0.2,
            color = 'blue',
            data = birth
            )

plt.title('npvis & bwght')
plt.ylabel('bwght')
plt.grid(True)


####father age and birthweight          ---- scatter!
plt.scatter(
            x = 'fage',
            y = 'bwght',
            alpha = 0.2,
            color = 'blue',
            data = birth
            )

plt.title('fage & bwght')              
plt.ylabel('bwght')
plt.grid(True)


####feduc and birthweight
plt.scatter(
            x = 'feduc',
            y = 'bwght',
            alpha = 0.2,
            color = 'blue',
            data = birth
            )

plt.title('feduc & bwght')
plt.ylabel('bwght')
plt.grid(True)


###cigs and bwght                ---- scatter!
plt.scatter(
            x = 'cigs',
            y = 'bwght',
            alpha = 0.2,
            color = 'blue',
            data = birth
            )

plt.title('cigs & bwght')
plt.ylabel('bwght')
plt.grid(True)


####drinks and bwght
plt.scatter(
            x = 'drink',
            y = 'bwght',
            alpha = 0.2,
            color = 'blue',
            data = birth
            )

plt.title('drink & bwght')
plt.ylabel('bwght')
plt.grid(True)
###if we get rid of the nondrinkers, we will have a nice descending line


###############################################################################
# Correlation Analysis
###############################################################################

df_corr = birth.corr().round(2)

print(df_corr)

df_corr.loc['bwght'].sort_values(ascending = False)


########################
# Correlation Heatmap
########################

# Using palplot to view a color scheme

sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize = (15, 15))

df_corr2 = df_corr.iloc[1:19, 1:19]

sns.heatmap(
            df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5
            )

plt.savefig('Birthweight Correlation Heatmap.png')
plt.show()


###############################################################################
###############################################################################
# Models
###############################################################################
###############################################################################

"""
This section explores models using OLS, KNN, Ridge and Lasso regressions. 
The best models selected are with variables: 
    
    Dependent   : bwght(Birthweight)
    Independent : mage(Mother Age grouped <=55 or >55),Cigs (cigarettes), 
                  Drink and average number of years of parents' education
    
More details in following python script regarding:
    - Variables used in models
    - Variables not used in models but good additional insights 
 
"""


###############################################################################
# Using OLS (bwght)
###############################################################################


'''
Dependent Variable:bwght (birthweight)
Independent Variable: cigs, drinks, mother's age (grouped <=55 or >55) and
                     average number of years of parents' education


Model Approach: 
1. Select significant variables and run an OLS regression 
2. Pick best model using R^2 , AIC and BIC


Comments:
    
- The best model (Model 1) explains 72% variation in birth weights and 
  predictors are significant at 95% confidence interval. 

- The model's probablity(F-statistics) is way below zero indicating
  explanatory power of all predictors together is significant at 5%.

- Cigs and drink negatively impacts birthweight with drinks impacting more 
  than cigs, however mothers younger than or equal 55 yrs have babies weighing 
  562 grams more than mother older than 55 yrs.

- As the average combined education of the parents increases by 1 year, the 
  baby born will be heavier by 24grams.

- Model 1 has the least AIC=2850 and BIC=2866, with highest R^2 = 72% 

- As R^2 is given more emphasis, we chose model 1 to be our best model.

'''


### Model 1: OLS with significant variables (Best Model)

lm_full = smf.ols(formula = """bwght ~ new_mage 
                                       + cigs 
                                       + drink
                                       + avg_combined_educ""",
                                       data = birth)

results = lm_full.fit()

print(results.summary())


### Model 2: OLS with significant variables 

lm_full = smf.ols(formula = """bwght ~ mage 
                                       + cigs 
                                       + drink""",
                                       data = birth)

results = lm_full.fit()

print(results.summary())


### Model 3: OLS with significant variables (cigarettes and drinks)

lm_full = smf.ols(formula = """bwght ~ cigs 
                                       + drink""", 
                                       data = birth)

results = lm_full.fit()

print(results.summary())


############################
# Additional : Using race
############################

"""
Rationale: Mother’s and Father’s race has an influence on baby’s weight. 

Dependent Variable:bwght (birthweight)
Independent Variable:new_race (Grouped races)

Data: We have 3 race categories: black, white and others. m represents mother,
f represents father. Total 6 variables.

Model Approach:
    
1.	Select 4 variables: mblck, mwhte, fblck, fwhte. Note this is sufficient
    and we do not need the moth and foth variable because it is redundant as 
    the others race can be determined from the 4 variables directly. 
    - Run OLS regression model – 4 variables are insignificant as the 
    p-value is more than 0.05.
         
2.	Group the variables by race for mother and father. 
    Eg. race_group_1 =  mblck+fblck.
    - Run OLS regression model for birthweight and race_group. As the p-value
    is more than 0.05, it is not significant. 
    
3.	Group the variables by same race (assigned 1) and multi-race (assigned 0). 
    - Run OLS regression model for birthweight and race_group. P-value is 
    0.015, hence it is significant. 

Comments:
    
-	If the mother and father are both of the same race, the baby born will on
    average have a birthweight 371 grams lower than than the baby whose mother
    and father were of different races. 
    
-	There are 176 babies born from parents of the same race and 20 babies born 
    from parents of different races. This may result in a low R-squared value.
    
-   The parent’s race does not come out in the final model selected by the team
    as in the presence of other variables, race becomes insignificant. 

""" 

### OLS for race_group by races

birth['new_race'].value_counts()

lm_full = smf.ols(formula = 'bwght ~ new_race', data = birth)

results = lm_full.fit()

print(results.summary())

### OLS for race_group by same race and multi-race 

birth['new_race_2'].value_counts()

lm_full = smf.ols(formula = 'bwght ~ new_race_2', data = birth)

results = lm_full.fit()

print(results.summary())


############################
# Additional : Using education
############################


"""
Rationale: More educated mother would take better care of herself and the baby,
hence reducing occurrence of low birth weight of babies. 
Education is linked to social economic status and the capability of taking
care during pregnancy. 

Dependent Variable:bwght (birthweight)
Independent Variable:avg_combined_educ (average of parents' education 
number of years)

Model Approach:
    
1.	Focusing on mother’s education alone
        - Run OLS regression model with birthweight and mother’s education : 
            Not significant
2.	Grouping mother’s education for education <= 12 years and >12 years
        - Run OLS regression model with birthweight and mother’s education : 
            Not significant
3.	Combining average of mother and father’s education: 
        - Run OLS regression model with birthweight and the average of 
        both mother’s and father’s education. It is significant at 90% 
        confidence interval with a p-value of 0.075. 

Comments:
    
- As the average education of parents increases by one year, the baby born
  will be 40 grams heavier. 


"""


# OLS for Average combined education of parents

lm_full = smf.ols(formula = 'bwght ~ avg_combined_educ',
                                     data = birth)

results = lm_full.fit()

print(results.summary())


##############################################################################
# Using OLS (new_bwght 0 and 1)
###############################################################################


"""
Rationale: Test if better models if different grouping by birthweight as 
0 and 1 (for babies with weight less than 2500 grams)

Comments: 
    
- Model 1 was selected as it has the highest R^2 of 30%.

- To avoid having a baby born with weight less than 2500 grams, the mother 
  must not smoke, must not drink and should have more prenatal visits. 
  For every additional prenatal visit, the probability of having a baby 
  whose weight is less than 2500 grams decreases by 0.01.

  
- Moreover the condition number is also slightly high at 65, hence there 
  might be some multicollinearity present. This can be evaluated with a 
  larger dataset.
  
"""


### Model 1: For significant variables

lm_birth = smf.ols(formula = """new_bwght ~ npvis 
                                            + cigs 
                                            + drink""",
                                            data = birth)

lm_birth_results = lm_birth.fit()

lm_birth_results.summary()


### Model 2: For significant variables 

lm_birth = smf.ols(formula = """new_bwght ~ new_npvis_2
                                            + cigs 
                                            + drink""",
                                            data = birth)

lm_birth_results = lm_birth.fit()

lm_birth_results.summary()


### Model 3: For significant variables 

lm_birth = smf.ols(formula = """new_bwght ~ cigs 
                                            + drink""",
                                            data = birth)

lm_birth_results = lm_birth.fit()

lm_birth_results.summary()

############################
# Additional : Using grouped mon_pre
############################


""" 
Rationale: There should be some releationship between birthweight and month 
prenatal care began.

Dependent Variable:new_bwght(Birthweight grouped)
Independent Variable:new_monpre(Month prenatal care began in 0-3 months)
    
Comments: 
    
- A mother who takes her prenatal vitamins within the first three months,
  has a probability of her baby weighing on average 20%  heavier than a 
  mother who takes her prenatal after the first trimester.
  
- The month prenatal care began does not come out in the final model selected
  by the team as in the presence of other variables, it becomes
  insignificant. 
    
    
"""


lm_full = smf.ols(formula = """new_bwght_2 ~ new_monpre_2""",
                                             data = birth)

results = lm_full.fit()
print(results.summary())

np.exp(-0.2196)


############################
# Additional : Using grouped npvis
############################

""" 
Rationale: There should be some releationship between birthweight and number
of visits to the doctor.

Dependent Variable:new_bwght(Birthweight grouped)
Independent Variable:new_npvis(# of prenatal visits to doctor <12 or >=12)
    
Comments: 
    
- Prenatal Visits stands significant at 95% confidence interval.
 
- If a mother paid less than or equal to  12 visits to doctor during her 
 pregnancy, her baby weighs 0.1% lower than a mother who pays more than 
 12 visits.
 
- The number of visits to the doctor does not come out in the final model
selected by the team as in the presence of other variables, it becomes
insignificant. 

"""

lm_full = smf.ols(formula = """new_bwght ~ new_npvis_2""",
                                           data = birth)

results = lm_full.fit()

print(results.summary())

np.exp(-0.0915)


###############################################################################
# MAPE
###############################################################################


def mean_absolute_percentage_error(y_true, y_pred): 

    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    return np.mean(np.abs((y_true - y_pred)/ y_true))*100


###############################################################################
# OLS Predictions
###############################################################################

### With 4 significant variables - but mother's age is grouped >55 or <=55

X = birth[['new_mage' ,
           'cigs' ,
           'drink',
           'avg_combined_educ']]

y = birth['bwght']

X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.10,
                                                    random_state = 508) 

# Training set 
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)

reg_all = LinearRegression()

reg_all.fit(X_train, y_train)

ols_y_pred = reg_all.predict(X_test)

print('Training Score', reg_all.score(X_train, y_train))
print('Testing Score:', reg_all.score(X_test, y_test))

OLS_train_score = (reg_all.score(X_train, y_train)).round(2)

OLS_test_score = (reg_all.score(X_test, y_test)).round(2)

diff_OLS_score = ((reg_all.score(X_train, y_train)) 
                   - (reg_all.score(X_test, y_test))).round(2)

mape_OLS = mean_absolute_percentage_error (y_test, ols_y_pred).round(2)

mae_OLS = mean_absolute_error(y_test, ols_y_pred)

mse_OLS = mean_squared_error(y_test,ols_y_pred)


###############################################################################
# Generalization using Train/Test Split
###############################################################################

X_tt  = birth.drop(['bwght',
                    'fmaps',
                    'omaps'],
                     axis = 1)

y_tt = birth.loc[:,'bwght']
 
X_train, X_test, y_train, y_test = train_test_split(
                                                    X_tt,
                                                    y_tt,
                                                    test_size = 0.10,
                                                    random_state = 508)

# Training set 
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)


""" 
Comments: From the train/test predictions below for KNN, Ridge and Lasso, 
the Ridge Regression was the best because it had the highest R^2 of 69% for 
the test accuracy scores. The difference was also the smallest at 0.03. 
KNN was not chosen because although it had a smaller difference of -0.02,
firstly it was overfitting the data and secondly the R^2 was smaller. 

"""


###############################################################################
# Forming a Base for Machine Learning with KNN
###############################################################################


"""
Model Approach: 
    1. Selecting all variables and running KNN train/test and identifying 
optimal number of neighbours
    2. Using optimal number of neighbours to run model
    
Comments: 
    - When all variables were selected, the R2 was 0.99 but this is not logical
     and this result was due to many insignficant factors present in the model.
    - Using the selected 4 variables , the KNN model ran selected 25 neighbours
      as the optimal 
    
"""
### With 4 significant variables - but mother's age is grouped

X = birth[['new_mage' , 'cigs' , 'drink', 'avg_combined_educ']]
y = birth['bwght']

X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y, 
                                                    test_size = 0.10, 
                                                    random_state = 508)

# Creating new lists to store training & test scores

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)

for n_neighbors in neighbors_settings:
 
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    training_accuracy.append(clf.score(X_train, y_train))

    test_accuracy.append(clf.score(X_test, y_test))


# Plotting the visualization
fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()


print(max(test_accuracy))
print(test_accuracy.index(max(test_accuracy))+1)

# Number of neighbours = 25

knn_reg = KNeighborsRegressor(algorithm='auto', n_neighbors=25)
knn_reg.fit(X_train, y_train)
knn_y_pred=knn_reg.predict(X_test)

#Accuracy
train_accuracy = knn_reg.score(X_train, y_train)
test_accuracy = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(train_accuracy)
print(test_accuracy)

# Comparing test and predicted 
compiled_y = pd.DataFrame({'Actual': y_test, 'Predicted': knn_y_pred.round(0)})
print(compiled_y)
rmse = np.sqrt(mean_squared_error(y_test, knn_y_pred))
mae = mean_absolute_error(y_test, knn_y_pred)
mse = mean_squared_error(y_test,knn_y_pred)

mape_KNN = mean_absolute_percentage_error (y_test, knn_y_pred).round(2)


KNN_train_score = (knn_reg.score(X_train, y_train)).round(2)
KNN_test_score = (knn_reg.score(X_test, y_test)).round(2)
diff_KNN_score = ((knn_reg.score(X_train, y_train)) 
               - (knn_reg.score(X_test, y_test))).round(2)



###############################################################################
# Ridge Regression
###############################################################################
"""
Ridge regression was used as it is a good technique to test for regression 
data that suffers from multicollinearity. As mentioned above, our best model 
has a condition number of 68 which is larger than 30, hence there is a slight
issue of multicollinearity. 

Comments
 - Good R2 (test_accuracy) using the 3 signficant variables of grouped 
   mother's age, cigs and drinks. 

"""


# With 4 significant variables - but mother's age is grouped >50 or <=50

X = birth[['new_mage' ,
           'cigs' ,
           'drink',
           'avg_combined_educ']]

y = birth['bwght']

X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.10,
                                                    random_state = 508)

ridge = Ridge (alpha = 0.1, normalize = True)

ridge.fit(X_train, y_train)

ridge_y_pred = ridge.predict(X_test)

print('Training Score', ridge.score(X_train, y_train))
print('Testing Score:', ridge.score(X_test, y_test))

ridge_train_score = (ridge.score(X_train, y_train)).round(2)

ridge_test_score = (ridge.score(X_test, y_test)).round(2)

diff_ridge_score = ((ridge.score(X_train, y_train)) 
                     - (ridge.score(X_test, y_test))).round(2)


mape_Ridge = mean_absolute_percentage_error (y_test, ridge_y_pred).round(2)

mae_Ridge = mean_absolute_error(y_test, ridge_y_pred).round(2)

mse_Ridge = mean_squared_error(y_test, ridge_y_pred).round(2)
 

###############################################################################
# Lasso Regression
###############################################################################

"""
Lasso regression is used to select the subset of variables as it is similar to
ridge regression but with a stronger penalty function. 

"""


# With 4 significant variables - but mother's age is grouped

X = birth[['mage' ,
           'cigs' ,
           'drink',
           'avg_combined_educ']]

y = birth['bwght']

X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.10,
                                                    random_state = 508)


lasso = Lasso(alpha = 1.9, normalize = True)

lasso.fit(X_train,y_train)

lasso_y_pred = lasso.predict(X_test)

lasso_coef = lasso.coef_

print(lasso_coef)


lasso.score(X_test, y_test)


print('Training Score', lasso.score(X_train, y_train))
print('Testing Score:', lasso.score(X_test, y_test))

lasso_train_score = (lasso.score(X_train, y_train)).round(2)

lasso_test_score = (lasso.score(X_test, y_test)).round(2)

diff_lasso_score = ((lasso.score(X_train, y_train)) 
                    - (lasso.score(X_test, y_test))).round(2)


mape_Lasso = mean_absolute_percentage_error (y_test, lasso_y_pred).round(2)

mae_Lasso = mean_absolute_error(y_test, lasso_y_pred).round(2)

mse_Lasso = mean_squared_error(y_test, lasso_y_pred).round(2)


###############################################################################
# Storing Model Predictions and Summary
###############################################################################

# Model Predictions
model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'KNN_Predicted': knn_y_pred,
                                     'OLS_Predicted': ols_y_pred,
                                     'Ridge_Predicted': ridge_y_pred,
                                     'Lasso_Predicted': lasso_y_pred})


KNN = pd.DataFrame({"KNN": [KNN_train_score,
                            KNN_test_score,
                            diff_KNN_score,
                            mape_KNN]})
    
OLS = pd.DataFrame({"OLS": [OLS_train_score,
                            OLS_test_score,
                            diff_OLS_score,
                            mape_OLS]})
    
Ridge = pd.DataFrame({"Ridge": [ridge_train_score,
                                ridge_test_score,
                                diff_ridge_score,
                                mape_Ridge]})
    
Lasso = pd.DataFrame({"Lasso": [lasso_train_score,
                                lasso_test_score,
                                diff_lasso_score,
                                mape_Lasso]})

    
# Model Scores
model_scores = pd.concat([KNN,
                          OLS,
                          Ridge,
                          Lasso],
                          axis = 1)

model_scores = model_scores.rename({0: "Training_score",
                                    1: "Test_score",
                                    2: "Difference",
                                    3: "Mape"},
                                    axis='index')

""" 
Comments: As seen from the models, KNN is overfitting the data and has low 
predictive accuracy. OLS and Ridge have the highest predictive
accuracy and as OLS has the least mape, we chose it as our best model. 
"""


# Extract model predictions for best model

best_model = pd.DataFrame({'Actual' : y_test,
                           'OLS_Predicted': ols_y_pred})

best_model.to_excel("Bwth_Best_Model_Predictions.xlsx", index = False)


""" Additional Model

Dependent Variable:bwght (Birthweight)
Independent Variable:fage,new_mage,cigs,drink,avg_combined_educ
(Monther age >55 or <=55)

Test score = 70.5%

We didn't chose it as our best model because father and mother age are highly 
correlated and both of them individually stand significant but not together.


X = birth[['new_mage' ,
               'cigs' ,
               'drink',
               'avg_combined_educ',
               'fage'
               ]]

y = birth['bwght']

X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.10,
                                                    random_state = 508)

ridge = Ridge (alpha = 0.1, normalize = True)

ridge.fit(X_train, y_train)

ridge_y_pred = ridge.predict(X_test)

print('Training Score', ridge.score(X_train, y_train))
print('Testing Score:', ridge.score(X_test, y_test))

ridge_train_score = (ridge.score(X_train, y_train)).round(2)

ridge_test_score = (ridge.score(X_test, y_test)).round(2)

diff_ridge_score = ((ridge.score(X_train, y_train)) 
                     - (ridge.score(X_test, y_test))).round(2)


mape_Ridge = mean_absolute_percentage_error (y_test, ridge_y_pred).round(2)

mae_Ridge = mean_absolute_error(y_test, ridge_y_pred).round(2)

mse_Ridge = mean_squared_error(y_test, ridge_y_pred).round(2)
"""














