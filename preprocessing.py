# Rescale data (between 0 and 1)
import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('2012-and-2016-presidential-elections/votes.csv')



#df = df[['POP010210','age65plus','AGE135214','VET605213','RHI625214','AGE295214','Poverty','Income','winner_16']]
#df = df[['POP010210','age65plus','SEX255214','White','Black','Edu_highschool','Edu_batchelors','Hispanic','INC110213','Poverty','winner_16']]
df = df[['votes_dem_2012','votes_gop_2012','NonEnglish','POP645213','Edu_highschool','Edu_batchelors','RHI425214','SEX255214', 'White', 'Black','VET605213', 'Hispanic','Poverty', 'Income','winner_16' ]]
df.dropna(inplace=True)


X = np.array(df.drop(['winner_16'],1))
X = preprocessing.scale(X)
y = np.array(df['winner_16'])
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])