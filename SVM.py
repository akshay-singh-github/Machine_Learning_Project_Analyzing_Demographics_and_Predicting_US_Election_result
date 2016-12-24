import pandas as pd
import quandl,math , datetime
import numpy as np
from scipy import sparse
from sklearn import preprocessing , cross_validation , svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from sklearn.feature_selection import VarianceThreshold

style.use('ggplot')

#from subprocess import check_output
#print(check_output(["ls", "2012-and-2016-presidential-elections"]).decode("utf8"))

votes = pd.read_csv('2012-and-2016-presidential-elections/votes.csv')
                                                                    
"""
featurelist
============
POP010210 : Population, 2010
AGE135214 : Persons under 5 years, percent, 2014
AGE295214 : Persons under 18 years, percent, 2014
AGE775214 : age65plus(given directly in votes.csv)
SEX255214 : Female persons, percent, 2014
RHI125214 : White(given directly)
RHI225214 : Black (given directly)
RHI325214 : American Indian and Alaska Native alone, percent, 2014
RHI425214 : Asian alone, percent, 2014
RHI525214 : Native Hawaiian and Other Pacific Islander alone, percent, 2014
RHI625214 : Two or More Races, percent, 2014
RHI725214 : Hispanic (given directly)
RHI825214 : White alone, not Hispanic or Latino, percent, 2014
POP715213 : Living in same house 1 year & over, percent, 2009-2013
POP645213 : Foreign born persons, percent, 2009-2013
POP815213 : NonEnglish (already gien based on language)
EDU635213 : Edu_highschool (already given)
EDU685213 : Edu_batchelors (already given)
VET605213 : Veterans, 2009-2013
LFE305213 : Mean travel time to work (minutes), workers age 16+, 2009-2013
HSG010214 : Housing units, 2014
HSG445213 : Homeownership rate, 2009-2013
HSG096213 : Housing units in multi-unit structures, percent, 2009-2013
HSG495213 : Median value of owner-occupied housing units, 2009-2013
HSD410213 : Households, 2009-2013
HSD310213 : Persons per household, 2009-2013
INC910213 : Income(already given)
INC110213 : Median household income, 2009-2013
PVY020213 : Poverty(given already)
BZA010213 : Private nonfarm establishments, 2013
BZA110213 : Private nonfarm employment,  2013
BZA115213 : Private nonfarm employment, percent change, 2012-2013
NES010213 : Nonemployer establishments, 2013
SBO001207 : Total number of firms, 2007
SBO315207 : Black-owned firms, percent, 2007
SBO115207 : American Indian- and Alaska Native-owned firms, percent, 2007
SBO215207 : Asian-owned firms, percent, 2007
SBO515207 : Native Hawaiian- and Other Pacific Islander-owned firms, percent, 2007
SBO415207 : Hispanic-owned firms, percent, 2007
SBO015207 : Women-owned firms, percent, 2007
MAN450207 : Manufacturers shipments, 2007 ($1,000)
WTN220207 : Merchant wholesaler sales, 2007 ($1,000)
RTN130207 : Retail sales, 2007 ($1,000)
RTN131207 : Retail sales per capita, 2007
AFN120207 : Accommodation and food services sales, 2007 ($1,000)
BPS030214 : Building permits, 2014
LND110210 : Land area in square miles, 2010
POP060210 : Density(already given)

Labels = c('Poverty', 'Income', 'Density', 'Bachelors Degree', 'Hispanic', 'Black', 'White', 'Obama', 'Romney', 'Clinton', 'Trump'),
identities = c('Poverty', 'Income', 'Density', 'Edu_bachelors', 'White', 'Black', 'Hispanic', 'Obama', 'Romney', 'Clinton', 'Trump')

"""

def CorrelationCoeff(X,Y):
    mu_x = np.mean(X)
    mu_y = np.mean(Y)
    N = len(X)
    r = (sum([ X[i]*Y[i] for i in range(N) ]) - N*mu_x*mu_y) \
        / (math.sqrt( sum(X**2)-N*(mu_x**2) )) \
        / (math.sqrt( sum(Y**2)-N*(mu_y**2) ))
    return r


############################################################################
#1 means trump
#0 means hillary

for i,row in enumerate(votes.iterrows()):
    if row[1].votes_gop_2016>row[1].votes_dem_2016:
        votes.loc[i,'winner_16'] = 1.0
    else:
        votes.loc[i,'winner_16'] = 0.0



#votest = votes [['votes_dem_2016','votes_gop_2016','winner_16','state_abbr','county_name','total_votes_2016','Edu_batchelors', 'White', 'Black', 'Hispanic','Poverty', 'Income', 'Density']]

#votesp = votes[['POP010210','age65plus','AGE135214','VET605213','RHI625214','AGE295214','Poverty','Income','winner_16']]
#votesp = votes[['POP010210','age65plus','SEX255214','White','Black','Edu_highschool','Edu_batchelors','Hispanic','INC110213','Poverty','winner_16']]
votesp = votes [['votes_dem_2012','votes_gop_2012','NonEnglish','winner_16','POP645213','Edu_highschool','Edu_batchelors','RHI425214','SEX255214', 'White', 'Black','VET605213', 'Hispanic','Poverty', 'Income' ]]
votesp.dropna(inplace=True)
Xp = np.array(votesp.drop(['winner_16'],1))
Xp = preprocessing.scale(Xp)
yp = np.array(votesp['winner_16'])
Xp_train, Xp_test , yp_train , yp_test = cross_validation.train_test_split(Xp,yp, test_size=0.2)
#print (len(Xp),len(yp))
#clfp = LinearRegression()
clfp = svm.SVC(kernel='poly')
#train
clfp.fit(Xp_train,yp_train)
#test
accuracyp = clfp.score(Xp_test,yp_test)
print (accuracyp)

#####################################################

