import numpy as np
from scipy.special import expit
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import time
#import psutil


class MostLayPers(object):

    def __init__(self, epochs=100, n_output=0, n_features=0,
                 l1=0.0, l2=0.0, alpha=0, n_hiden=30,
                 decrease_const=0.0, eta=0.01, shuffle=True,
                 minibatches=1, random_state=0):

        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hiden = n_hiden
        self.flag_plt = 1
        self.w1, self.w2 = self._initialize_weight()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self, y, k):

        """
        Преобразование метки в прямую кодировку

        :param y:           массив, форма[n_samples]
        :param k:           узнать
        :return onehot:     массив, форма=(n_labels, n_samples)-кортеж
        """

        onehot = np.zeros((k, y.shape[1]))#изменено y.shape[0]===>y.shape[1]
        for idx, val in np.ndenumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weight(self):
        """ Инициализацияб весов"""

        w1 = np.random.uniform(-1, 1, size=self.n_hiden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hiden, self.n_features + 1)  # разобраться с инициализацией весов

        w2 = np.random.uniform(-1, 1, size=self.n_output * (self.n_hiden + 1))
        w2 = w2.reshape(self.n_output, self.n_hiden + 1)

        return w1, w2

    def _sigmoid(self, z):

        """Вычеслить логистическую функцию
        Использует функцию scipy.special.expit,
        чтобы избежать ошибки переполнения для очень малых входных значений z."""

        return expit(z)

    def sigmoid_gradient(self, z):

        """Вычислить градиент логистической функции (4)"""
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def add_bias(self, X, how='column'):
        """Добавление в массив коэф смещения
        (столбец или стороку из едениц) в нулевом индексе"""

        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('how shoud be column or row')
        return X_new

    def _feedforward(self, X, w1, w2):
        """ Вычисление шага прямого распространения сигнала

        :param      X: массивб форма=[n_samples, n_features]
                        Входной слой с исходными признаками
        :param      w1:массив форма = [n_hiden_units, n_features]
                        Матрица весовых коэффициентов для "входной слой->скрытый слой"
        :param      w2:  массив форма = [n_hiden_units, n_features]
                        Матрица весовых коэффициентов для "скрытый слой->выходной слой"

        :return:    a1:массив, форма=[n_samples, n_features]
                        Входные значения с узлом смещения.
                    z2:массив, форма=[[n_hidden, n_samples]
                       Чистый вход скрытого слоя.
                    a2:массив, форма=[n_samples, n_samples]
                        Активация скрытого слоя.
                    z3:массив, форма=[n_output_units, n_samples]
                        Чистый вход выходного слоя.
                    a3:массив, форма=[n_output_units, n_samples]
                        Активация выходного слоя.

        """

        a1 = self.add_bias(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self.add_bias(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)

        return a1, z2, a2, z3, a3

    def _L2_reg(self, lamb_, w1, w2):
        ''' Вычеслить L1 регуляризацию'''

        return (lamb_ / 2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

    def _L1_reg(self, lamb_, w1, w2):

        return (lamb_ / 2.0) * (np.abs(w1[:, 1:].sum()) + np.abs(w2[:, 1:]).sum())

    def get_cost(self, y_enc, output, w1, w2):
        """ Вычислить функцию стоимости

        :param      y_enc: массив, форма=[n_labels, n_samples]
                            Прямокодирование метки
        :param      output: массив, форма=[n_output_units, n_samples]
                            Активация выходного слоя(прямое распространение
        :param      w1: массив форма = [n_hiden_units, n_features]
                        Матрица весовых коэффициентов для "входной слой->скрытый слой"
        :param      w2:  массив форма = [n_hiden_units, n_features]
                        Матрица весовых коэффициентов для "скрытый слой->выходной слой"

        :return:    cost: float
                        Регуляризованная стоимость
        """
        #y_enc=y_enc.T
        term1 = -y_enc * (np.log(output))
        term2 = (1 - y_enc) * np.log(1 - output)

        cost = np.sum(term1 - term2)

        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)

        cost = cost + L1_term + L2_term

        #############через экспоненту##########
        term3=np.log(1+np.exp(-y_enc*output))
        cost2=np.sum(term3)
        cost2=cost2+L1_term + L2_term
        ######################################
        return cost2

    def _get_gradient(self, a1, a2, a3, z2, z3, y_enc, w1, w2):
        """Вычислить шаг градиента, используя обратное распространение

        :param      a1: массив, форма=[n_samples, n_features]
                        Входные значения с узлом смещения.
        :param      a2: массив, форма=[n_hidden+1, n_samples]
                        Активация скрытого слоя.
        :param      a3: массив, форма=[n_output_units, n_samples]
                        Активация выходного слоя.
        :param      z2: массив, форма=[[n_hidden, n_samples]
                        Чистый вход скрытого слоя.
        :param      y_enc: массив, форма=[n_labels, n_samples]
                            Прямокодирование метки
        :param      w1: массив форма = [n_hiden_units, n_features]
                        Матрица весовых коэффициентов для "входной слой->скрытый слой"
        :param      w2: массив форма = [n_hiden_units, n_features]
                        Матрица весовых коэффициентов для "скрытый слой->выходной слой"

        :return: grad1: массив, форма=[n_output_units, n_future]
                        Градиент матрицы весовых коэффициентов w1
                 grad2: массив, форма=[n_output_units, n_hiden_units]
                        Градиент матрицы весовых коэффициентов w2
        """

        # Обратное распространение
        # z2 = self.add_bias(z2, how='row')
        # sigma3_0 = (a3 - y_enc)  # (1)
        # sigma3=sigma3_0.dot(self.sigmoid_gradient(z2.T))
        # sigma2 = w2.T.dot(sigma3) * self.sigmoid_gradient(z2)#*self.sigmoid_gradient(a1)  # ->sigmoid_gradient(4)
        # sigma2 = sigma2[1:, :]
        # grad1 = sigma2.dot(a1)
        # grad2 = sigma3.dot(a2.T)
        """Вероятно пропущенно в рашке ссылаясь на индуса:
         для sigma3: sigmoid_gradient(z2)
         для sigma2: sigmoid_gradient(a1)"""

        ######################################
        sigma3 = (a3 - y_enc)*self.sigmoid_gradient(z3)
        z2 = self.add_bias(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self.sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        #a2 = a2 * self.sigmoid_gradient(z2)
        grad2 = sigma3.dot(a2.T)

        #sig_z3=self.sigmoid_gradient(z3)
        #result=sigma3*sig_z3
        #np.dot(a2, sig_z2.T)

        ######################################

        # Регуляризовать
        grad1[:, 1:] += self.l2 * w1[:, 1:]
        grad1[:, 1:] += self.l1 * np.sign(w1[:, 1:])
        grad2[:, 1:] += self.l2 * w2[:, 1:]
        grad2[:, 1:] += self.l1 * np.sign(w2[:, 1:])

        return grad1, grad2

    def fit(self, X, y, print_progress=False, visual_plot=False):
        """

        :param      X: массивб форма=[n_samples, n_features]
                        Входной слой с исходными признаками
        :param      y: массив, форма[n_samples]
                        Целевые метки класса
        :param      print_progress: bool[default:Falase]
                      Распечатывает ход работы в виде
                      числа эпох на устройстве stderr

        :return: self
        """

        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        n = np.array([1])
        n2 = np.array([0])
        flag=0

        plt.rcParams['animation.html'] = 'jshtml'
        if visual_plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.show()
            x_v, y_v = [], []


        for i in range(self.epochs):
            # адфптивный темп обучения
            self.eta /= (1 + self.decrease_const * i)

            if print_progress:
                sys.stderr.write('\rEpoha %d/%d' % (i + 1, self.epochs))
                sys.stderr.flush()
            if self.shuffle:
                idx = np.random.permutation(y_data.shape[1])#зменили индекс на 1
                #X_data, y_data = X_data[idx], y_data[idx[:, idx]]
                #X_data_fit = np.array(y_data.shape[1])
                print("idx[0]=")
                print(idx[0])

                ####добавили изменение шага######
            if i==700 and flag==0:
                self.l1=0.0007
                flag=1

                #################################
            # mini = np.array_split(range(
            #     y_data.shape[1]), self.minibatches)#зменили индекс на 1
            # Array_split-Создание массива с подмассивами


            for n[0] in range(2000):# изменение idx.shape[0]==>>2000
                # прямое распространение
                """ПРОБЛЕМА: ЗАЦИКЛИТЬ ТАК ЧТОБЫ ЦИКЛ ПРОБЕГЛАСЯ ПО МАССИВУ В СЛУЧАЙНОМ ПОРЯДКЕ X И Y
                В СЛУЧАЙНОМ ПОРЯДКЕ"""


                a1, z2, a2, z3, a3 = self._feedforward(X_data[idx[n]], self.w1, self.w2)#заменая X_data[idx]->X_data[idx[n]]

                cost = self.get_cost(y_enc[:, idx[n]],#заменая Idx->0
                                     output=a3,
                                     w1=self.w1,
                                     w2=self.w2)
                #self.cost_.append(cost)

                # вычислить градиент методом обратного распространения
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2,
                                                  a3=a3, z2=z2, z3=z3,
                                                  y_enc=y_enc[:, idx[n]],#заменая Idx->0
                                                  w1=self.w1, w2=self.w2)

                # # начало оценки градиента
                # grad_dif=self.estimate_grade( X=X[idx], y_enc=y_enc[:, idx],
                #                      w1=self.w1, w2=self.w2,
                #                      epsilon=1e-5, grad1=grad1, grad2=grad2)
                # if grad_dif<= 1e-7:
                #     print('Успешно %s' % grad_dif)
                # elif grad_dif<= 1e-4:
                #     print ('Предупреждение %s' % grad_dif)
                # else:
                #     print ('ПРОБЛЕМА %s' % grad_dif)
                # # Конец оценки градиента

                # обновить веса
                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev = delta_w1
                delta_w2_prev = delta_w2

            if visual_plot:
                x_v.append(i)
                y_v.append(cost)
                ax.plot(x_v, y_v, color='b')
                # fig.canvas.draw()
                ax.set_xlim(left=max(0, i - 100), right=i + 100)
                #ax.set_ylim(top=max(0, cost + 0.2), bottom=cost -0.05)
                # time.sleep(0.00001)
                fig.show()



        return self

    def estimate_grade(self, X,y_enc, w1, w2, epsilon, grad1,grad2):
        '''

        :param      X: массивб форма=[n_samples, n_features]
                        Входной слой с исходными признаками
        :param      y: массив, форма[n_samples]
                        Целевые метки класса
        :param      y_enc: прямое кодирование
        :param      w1: массив форма = [n_hiden_units, n_features]
                        Матрица весовых коэффициентов для "входной слой->скрытый слой"
        :param      w2: массив форма = [n_hiden_units, n_features]
        :param epsilon: ?
        :param grad1:   ?
        :param grad2:   ?
        :return:    relative_error : float  Относительная ошибка между численно аппроксимированными
                    градиентами и градиентами обратного распространения.
        '''

        num_grad1=np.zeros(np.shape(w1))
        epsilon_ary1=np.zeros(np.shape(w1))

        for i in range(w1.shape[0]):
            for j in range(w1.shape[1]):
                epsilon_ary1[i,j]=epsilon
                a1, z2, a2, z3, a3 = self._feedforward(X,
                                                       w1-epsilon_ary1,
                                                       w2)
                cost1=self.get_cost(y_enc,
                                    a3,
                                    w1 - epsilon_ary1,
                                    w2)
                a1, z2, a2, z3, a3 = self._feedforward(X,
                                                       w1 + epsilon_ary1,
                                                       w2)
                cost2=self.get_cost(y_enc,
                                    a3,
                                    w1 + epsilon_ary1,
                                    w2)
                num_grad1[i,j]=(cost2-cost1)/(2*epsilon)
                epsilon_ary1[i,j]=0
            print("i_w1=")
            print(i)

        num_grad2 = np.zeros(np.shape(w2))
        epsilon_ary2 = np.zeros(np.shape(w2))

        for i in range(w2.shape[0]):
            for j in range(w2.shape[1]):
                epsilon_ary1[i,j]=epsilon
                a1, z2, a2, z3, a3 = self._feedforward(X,
                                                       w1-epsilon_ary1,
                                                       w2)
                cost1=self.get_cost(y_enc,
                                    a3,
                                    w1 - epsilon_ary1,
                                    w2)
                a1, z2, a2, z3, a3 = self._feedforward(X,
                                                       w1 + epsilon_ary1,
                                                       w2)
                cost2=self.get_cost(y_enc,
                                    a3,
                                    w1 + epsilon_ary1,
                                    w2)
                num_grad2[i,j]=(cost2-cost1)/(2*epsilon)
                epsilon_ary2[i,j]=0
            print("i_w1=")
            print(i)
        num_grad=np.hstack((num_grad1.flatten(),
                            num_grad2.flatten()))# hstack-склеивает массивы
                                                 # flaten-сплющить в одно измерение

        grad=np.hstack((grad1.flatten(),
                        grad2.flatten()))

        norm1=np.linalg.norm(num_grad-grad)
        norm2=np.linalg.norm(num_grad)
        norm3=np.linalg.norm(grad)
        relative_error=norm1/(norm2+norm3)
        return relative_error




    def predict(self, X):
        """

        :param      X: массив форма=[n_samples, n_features]
                        Входной слой с исходными признаками

        :return:    y_pred : массив, форма = [n_samples]
                        Идентифицированные метки классов.
        """

        if len(X.shape) != 2:
            raise AttributeError(
                'X должен быть массивом [n_samples, n_features].\n'
                'Используйте X[:,None] для 1-признаковой классификации,'
                '\плибо X[[i]] для 1-точечной классификации'
            )

        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(a3, axis=0)
        return y_pred
