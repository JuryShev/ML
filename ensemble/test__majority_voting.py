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
import pandas as pd
from sklearn.svm import SVC  # метод опорных векторов
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn. ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data'
#datasets.load_iris()

iris=pd.read_csv(url,header=None)
iris=iris.drop(iris[iris[:][34]=='?'].index)
X=iris.loc[:, 2:]
y=iris.loc[:, 1]
#X,y= iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y_tr=le.fit_transform(y)

#________инициалтзация классификаоров________________________
clf1=LogisticRegression(penalty='l1', C=13, random_state=0)
clf1_1=SVC(random_state=0, C=13, probability=True, kernel='linear')
clf1_1_adaBoost=AdaBoostClassifier(base_estimator=clf1_1, n_estimators=600, learning_rate=1,random_state=0)
clf2=DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
clf2_adaBoost=AdaBoostClassifier(base_estimator=clf2, n_estimators=200, learning_rate=0.1,random_state=0)
clf3=KNeighborsClassifier(n_neighbors=2, p=3, metric='minkowski')

pipe1=Pipeline([('sc', StandardScaler()),
            ('pca', PCA(n_components=12)),
              ('clf', clf1)])
pipe2=Pipeline([('sc', StandardScaler()),
                ('clf', clf1_1)])
pipe3=Pipeline([('sc', StandardScaler()),
                ('clf', clf3)])
pipe4=Pipeline([('sc', StandardScaler()),
                ('clf', clf2_adaBoost)])

X_train, X_test, y_train, y_test =train_test_split(X, y_tr, test_size=0.30, random_state=1)
                                                         #процентное содержание
                                                         #для метки test
mv_clf=MajorityVoteClassifier(classifiers=[pipe1,pipe2,pipe4], )
mv_clf.fit(X_train,y_train)
scores_val=cross_val_score(estimator=mv_clf,X=X_test, y=y_test, cv=10, scoring='roc_auc')
print("ROC AUC: %0.2f (+/- %0.2f) [%s]",(scores_val.mean(), scores_val.std()))
all_clf = [pipe1, pipe2, pipe3,pipe4, mv_clf]
clf_labels=['LogReg','SVC','KN','Tree','MVC']
gr=mv_clf.get_params
#____________________________________________________________
#param_grid2=[{'pipeline-1__clf__C': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100, 1000],
#             'decisiontreeclassifier_max_depth':[1,2]}]
            #  'pca__n_components': comp_PCA}]
#grid = GridSearchCV(estimator=mv_clf,
#                    param_grid=param_grid2,
#                    cv=10,
#                    scoring='roc_auc')
#grid.fit(X_train,y_train)
#print('\ngrid', grid.best_score_)
#___________________поиск по сетке___________________________

#____________________________________________________________
#____________________________________________________________
#____________________________________________________________

#____________________Оценка методов__________________________________________
colors=['black', 'orange', 'blue', 'green', 'red']

linestyles=[':','--','-.','-','-']
print(all_clf[1])

for i in range(len(all_clf)):
    y_pred=all_clf[i].fit(X_train, y_train).predict_proba(X_test)[:,1]

    fpr, tpr, thresholds=roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc=auc(x=fpr,y=tpr)
    plt.plot(fpr,tpr,color=colors[i],linestyle=linestyles[i], label='%s (auc = %0.2f)'%(clf_labels[i],roc_auc))
print('\nget_params', mv_clf.get_params)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1], linestyle='--',color='gray',linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid ()
plt.xlabel('Доля ложноположительных')
plt.ylabel('Доля истинно положительных')
plt.show()