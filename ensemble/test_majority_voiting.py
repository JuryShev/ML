from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from majority_voting import MajorityVoteClassifier
import numpy as np


iris = datasets.load_iris()
X,y= iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
у = le.fit_transform(y)

#________инициалтзация классификаоров________________________
clf1=LogisticRegression(penalty='l2', C=0.001, random_state=0)
clf2=DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3=KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1=Pipeline([('sc', StandardScaler()),
                ('clf', clf1)])
pipe2=Pipeline([['sc', StandardScaler()],
              ['clf', clf2]])
pipe3=Pipeline([['sc', StandardScaler()],
              ['clf', clf3]])
mv_clf=MajorityVoteClassifier(clssifiers=[pipe1,pipe2,pipe3])
#____________________________________________________________