import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


def k_nearest_neighbors(data, predict, k=3):
	if len(data)>=k:
		warnings.warn('K is set to a value less than total voting groups!')
	distances=[]
	for group in data:
		for features in data[group]:
			euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean_distance, group])
	votes= [i[1] for i in sorted(distances)[:k]]
	vote_result=Counter(votes).most_common(1)[0][0]
	return vote_result

df=pd.read_csv('2012-and-2016-presidential-elections/votes.csv')
for i,row in enumerate(df.iterrows()):
    if row[1].votes_gop_2016>row[1].votes_dem_2016:
        df.loc[i,'winner_16'] = 1.0
    else:
        df.loc[i,'winner_16'] = 0.0

#df = df[['POP010210','age65plus','AGE135214','VET605213','RHI625214','AGE295214','Poverty','Income','winner_16']]
#df = df[['POP010210','age65plus','SEX255214','White','Black','Edu_highschool','Edu_batchelors','Hispanic','INC110213','Poverty','winner_16']]
df = df[['votes_dem_2012','votes_gop_2012','NonEnglish','POP645213','Edu_highschool','Edu_batchelors','RHI425214','SEX255214', 'White', 'Black','VET605213', 'Hispanic','Poverty', 'Income','winner_16' ]]
df.dropna(inplace=True)
#print(df)

full_data = df.astype(float).values.tolist()

random.shuffle(full_data)
test_size = 0.2
train_set={0:[], 1:[]}
test_set= {0:[], 1:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
        train_set[i[-1]].append(i[:-1])

for i in test_data:
	test_set[i[-1]].append(i[:-1])

correct = 0
total =0

for group in test_set:
	for data in test_set[group]:
		vote = k_nearest_neighbors(train_set, data, k=5)
		if group==vote:
			correct+=1
			
		total+=1

acc=(correct*1.0)/total		
print (acc)
