from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
np.random.seed(0)

iris=load_iris()
print(iris)
#Creating a dataframe
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df.head

df['species']=pd.Categorical.from_codes(iris.target,iris.target_names)
df.head

#SPLITTING THE DATA INTO TEST AND TRAIN
df['is_train']=np.random.uniform(0,1,len(df))<=.75
df.head

#Creating train and test
train,test = df[df['is_train']==True],df[df['is_train']==False]
print('Number of observations in train set:',len(train))
print('Number of observations in test set:',len(test))

#Column names
features=df.columns[:4]
features

#Factorizing the type of species
y=pd.factorize(train['species'])[0]
y

#Creating random forest classifier
clf=RandomForestClassifier(n_jobs=2, random_state=0)
#Training the classifier
clf.fit(train[features],y)

#Applying the rfc on test data
clf.predict(test[features])

#mapping names
preds=iris.target_names[clf.predict(test[features])]
preds[0:25]

#Actual species for first 5 observations
test['species'].head()

#Confusion matrix
pd.crosstab(test['species'],preds, rownames=['Actual Species'], colnames=['Predicted Species'])