import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix  # матрица несоответствий
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
# LabelEncoder-преобразует символьные метки в численные
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
# LabelEncoder-кодирует коткгориальные данные в вид приемлемый для обучения

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # метод опорных векторов
from sklearn.pipeline import _name_estimators
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import operator
from sklearn.externals import six

class MajorityVoteClassifier (BaseEstimator, ClassifierMixin):
    """Ансамблевый классификатор на основе мажоритарного голосования

    Парметры:       classlabel: массивоподобная форма=[n_classifier]
                                Список классификатора для ансамбля

                    vote : str, {'classlabel', 'probability'}
                            По умолчанию classlabel.
                            Если метка класса <classlabel',
                            то прогноз основывается на argmax меток классов.
                            В противном случае если <probability',
                            то для прогноза метки класса используется argmax
                            суммы вероятностей (рекомендуется для откалиброванных классификаторов).
                    weights: массивоподобный, форма = [n_classifiers]
                            Факультативно, по умолчанию: None
                            Если предоставлен список из значений 'int' либо 'float', то Классификаторы взвешиваются по важности;
                            Если 'weights=None', то используются равномерные веса


    """

    def __init__(self, classifiers,
                 vote='classlabel', weights=None):
        self.classifiers=classifiers
        self.named_classifirs={key: value for
                               key, value in _name_estimators(classifiers) }
        self.vote =vote
        self.weights=weights

    def fit ( self, X,y):
        """ Выполнить подгонку классификатора

        Параметры:     Х: массивоподобный, разряженная матрица
                       форма=[n_samples, n_features]
                       Матрица с тренировачными образцами

                       y: массивоподобный,
                          форма = [n_samples]
                          Вектор целевых меток классов
        Возвращает:

        self: объект

        Использовать LabelEncoder, чтобы гарантировать, что
        метки классов начинаются с 0, что важно для
        вызова np.argmax в self.predict
        """

        self.labelnc_ =LabelEncoder()
        self.labelnc_.fit(y)
        self.classes=self.labelnc_.classes_# присвоение классификайций
        self.classifiers_=[]
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.labelnc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):

        '''
        Параметры:     Х: массивоподобный, разряженная матрица
                       форма=[n_samples, n_features]
                       Матрица с тренировачными образцами
        Возвращает:    maj_vote: массивоподобный, форма=[n_samles]
                    спрогназированные метки класса

        '''

        if self.vote=='probability':
            maj_vote=np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions=np.asarray([clf.predict(X)
                                for clf in self.classifiers_]).T
            # asarray- работает с оригиналом не создавая копию
            #96-97 предсказывает н на с помощью инструментов записанных в fit, 79 строка

            maj_vote=np.apply_along_axis(lambda x: np.argmax(np.bincount(x,
                                                                         weights=self.weights)),
                                         axis=1,
                                         arr=predictions)
            # np.bincount-определяет как чато повторяются значения int-положительные

        maj_vote=self.labelnc_.inverse_transform(maj_vote)
        # inverse_transform-трансформирует метку в исходную кодировку метод LabelEncoder


        return maj_vote
    def predict_proba(self, X):
        ''' Спрогнозировать вероятность классов для Х

            Параметры       X:  { массивоподобный, разреженная матрица},
                            форма = [n_samples, n_features]
                            Тренировочные векторы,
                            где n_samples - число образцов
                            n_features - число признаков.

            Возвращает:     avg_proba : массивоподобный,
                            форма = [n_samples, n_classes]
                            Взвешенная средняя вероятность
                            для каждого класса в расчете на образец.

        '''

        probas=np.asarray([clf.predict_proba(X)
                       for clf in self.classifiers_])
        avg_proba=np.average(probas,
                             axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        '''Получить имена параметров классификатора для GridSearch"'''
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out=self.named_classifirs.copy()
            for name, step in six.iteritems(self.named_classifirs): #six.iteritems-Возвращает итератор по элементам словаря.
                for key, value in six.iteritems(step.get_parms(deep=True)):
                    out['%s_%s' % (name, key)] = value
            return out




