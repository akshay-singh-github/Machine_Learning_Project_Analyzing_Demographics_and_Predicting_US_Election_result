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


def CorrelationCoeff(X,Y):
    mu_x = np.mean(X)
    mu_y = np.mean(Y)
    N = len(X)
    r = (sum([ X[i]*Y[i] for i in range(N) ]) - N*mu_x*mu_y) \
        / (math.sqrt( sum(X**2)-N*(mu_x**2) )) \
        / (math.sqrt( sum(Y**2)-N*(mu_y**2) ))
    return r


for i,row in enumerate(votes.iterrows()):
    if row[1].votes_gop_2016>row[1].votes_dem_2016:
        votes.loc[i,'winner_16'] = 1.0
    else:
        votes.loc[i,'winner_16'] = 0.0

votes = votes [['votes_dem_2012','votes_gop_2012','Trump','Clinton','NonEnglish','winner_16','POP645213','Edu_highschool','Edu_batchelors','RHI425214','SEX255214', 'White', 'Black','VET605213', 'Hispanic','Poverty', 'Income' ]]
votes.dropna(inplace=True)
#POP645213 : Foreign born persons
plt.scatter(votes['POP645213'], votes['Trump'], s=8, c='r', label='X-Axis: % Foreign born persons vs. Y-Axis: % Votes for Rep. in 2016')
plt.scatter(votes['POP645213'], votes['Clinton'], s=8, c='b', label='X-Axis: % Foreign born persons vs. Y-Axis: % Votes for Dem. in 2016') 
plt.legend()
X = votes.loc[:,'POP645213'].values
Y = votes.loc[:,'Trump'].values
r = CorrelationCoeff(X,Y)
print('Correlation Coefficient for Foreign born persons Vs Republicans  = ' + str(r))

Z = votes.loc[:,'POP645213'].values
W = votes.loc[:,'Clinton'].values
s = CorrelationCoeff(Z,W)
print('Correlation Coefficient for Foreign born persons Vs Democrats  = ' + str(s))


plt.show()
