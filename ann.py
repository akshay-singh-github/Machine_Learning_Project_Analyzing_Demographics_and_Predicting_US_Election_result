import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, ensemble, tree, neural_network
import pandas as pd

#df = pd.read_csv('votes2.csv')

df=pd.read_csv('2012-and-2016-presidential-elections/votes.csv')
for i,row in enumerate(df.iterrows()):
    if row[1].votes_gop_2016>row[1].votes_dem_2016:
        df.loc[i,'winner_16'] = 1.0
    else:
        df.loc[i,'winner_16'] = 0.0

#df = df[['X','POP010210','age65plus','AGE135214','VET605213','AGE295214','winner_16']]
#df = df[['POP010210','age65plus','SEX255214','White','Black','Edu_highschool','Edu_batchelors','Hispanic','INC110213','Poverty','winner_16']]
df = df[['votes_dem_2012','votes_gop_2012','NonEnglish','POP645213','Edu_highschool','Edu_batchelors','RHI425214','SEX255214', 'White', 'Black','VET605213', 'Hispanic','Poverty', 'Income','winner_16' ]]
df.dropna(inplace=True)



X = np.array(df.drop(['winner_16'],1))
X = preprocessing.scale(X)
y = np.array(df['winner_16'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neural_network.MLPClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print (accuracy)
