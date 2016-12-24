"""
# Feature Importance with Extra Trees Classifier
import pandas
import numpy
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
url = '/Users/Aunsh/PycharmProjects/votesimpf.csv'
names = ['votes_dem_2012','votes_gop_2012','AGE135214','AGE295214','age65plus','SEX255214','White','Black','RHI325214','RHI425214','RHI525214','RHI625214','Hispanic','RHI825214','POP715213','POP645213','NonEnglish','Edu_highschool','Edu_batchelors','VET605213','LFE305213','HSG010214','HSG445213','HSG096213','HSG495213','HSD410213','HSD310213','Income','INC110213','Poverty','BZA010213','BZA110213','NES010213','SBO001207','SBO315207','SBO115207','SBO215207','SBO515207','SBO415207','SBO015207','MAN450207','WTN220207','RTN130207','RTN131207','AFN120207','BPS030214','LND110210','Density','Winner']
dataframe = pandas.read_csv(url,usecols=names)
array = dataframe.values
X = array[:,0:49]
Y = array[:,49]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)


# Rescale data (between 0 and 1)
import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler
url = '/Users/Aunsh/PycharmProjects/preprocesseddata.csv'
names = ['votes_dem_2012','votes_gop_2012','AGE135214','AGE295214','age65plus','SEX255214','White','Black','RHI325214','RHI425214','RHI525214','RHI625214','Hispanic','RHI825214','POP715213','POP645213','NonEnglish','Edu_highschool','Edu_batchelors','VET605213','LFE305213','HSG010214','HSG445213','HSG096213','HSG495213','HSD410213','HSD310213','Income','INC110213','Poverty','BZA010213','BZA110213','NES010213','SBO001207','SBO315207','SBO115207','SBO215207','SBO515207','SBO415207','SBO015207','MAN450207','WTN220207','RTN130207','RTN131207','AFN120207','BPS030214','LND110210','Density', 'BZA115213','Winner']
dataframe = pandas.read_csv(url,usecols=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:49]
Y = array[:,49]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])



# Normalize data (length of 1)
from sklearn.preprocessing import Normalizer
import pandas
import numpy
url = '/Users/Aunsh/PycharmProjects/preprocesseddata.csv'
names = ['votes_dem_2012','votes_gop_2012','AGE135214','AGE295214','age65plus','SEX255214','White','Black','RHI325214','RHI425214','RHI525214','RHI625214','Hispanic','RHI825214','POP715213','POP645213','NonEnglish','Edu_highschool','Edu_batchelors','VET605213','LFE305213','HSG010214','HSG445213','HSG096213','HSG495213','HSD410213','HSD310213','Income','INC110213','Poverty','BZA010213','BZA110213','NES010213','SBO001207','SBO315207','SBO115207','SBO215207','SBO515207','SBO415207','SBO015207','MAN450207','WTN220207','RTN130207','RTN131207','AFN120207','BPS030214','LND110210','Density', 'BZA115213','Winner']
dataframe = pandas.read_csv(url,usecols=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:49]
Y = array[:,49]
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(normalizedX[0:5,:])
"""
# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
# load data
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
scaler1 = Normalizer().fit(rescaledX)
normalizedX = scaler1.transform(rescaledX)
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(normalizedX, y)
# summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)

"""
# Feature Extraction with RFE
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
url = '/Users/Aunsh/PycharmProjects/votesimpf.csv'
names = ['votes_dem_2012','votes_gop_2012','AGE135214','AGE295214','age65plus','SEX255214','White','Black','RHI325214','RHI425214','RHI525214','RHI625214','Hispanic','RHI825214','POP715213','POP645213','NonEnglish','Edu_highschool','Edu_batchelors','VET605213','LFE305213','HSG010214','HSG445213','HSG096213','HSG495213','HSD410213','HSD310213','Income','INC110213','Poverty','BZA010213','BZA110213','NES010213','SBO001207','SBO315207','SBO115207','SBO215207','SBO515207','SBO415207','SBO015207','MAN450207','WTN220207','RTN130207','RTN131207','AFN120207','BPS030214','LND110210','Density','Winner']
dataframe = pandas.read_csv(url,usecols=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_
"""