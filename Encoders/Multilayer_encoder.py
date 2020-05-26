import tensorflow as tf
import numpy as np
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt

n_inputs=28*28
n_hidden_1=300
n_hidden_2=150
n_hidden_3=n_hidden_1
n_outputs=n_inputs

learning_rate=0.001
l2_reg=0.0001

n_epochs=10
batch_size=30

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def shuffle_batch(X, y, n_batches):
    rnd_idx = np.random.permutation(len(X))
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
training = tf.placeholder_with_default(False, shape=(), name='training')# необходим чтобы передать  tf.layers.batch_normalization
                                                                        # среднее смещение определять по сему набору данных или по пакету
                                                                        # False-весь набор данных, True-пакет

reset_graph()

X=tf.placeholder(tf.float32, [None, n_inputs])
he_init=tf.contrib.layers.variance_scaling_initializer() #initializer Xavier for wight in range sqrt(6/n_outputs+n_inputs)
l2_regularizer=tf.contrib.layers.l2_regularizer(l2_reg)

						# transive method
my_dense_layer=partial(tf.layers.dense, activation=tf.nn.elu,
					   kernel_initializer=he_init,
					   kernel_regularizer=l2_regularizer)
"""
Partial это класс от функции всшего порядка functools.
Partial Вы можете использовать для создания новой функции
с частичным приложением аргументов и ключевых слов, которые вы передаете.
Пример:
from functools import partial

def add(x, y):
    return x + y
 
def multiply(x, y):
    return x * y
 
def run(func):
    print(func())
 
def main():
    a1 = partial(add, 1, 2)
    m1 = partial(multiply, 5, 8)
    run(a1)
    run(m1)
	
if __name__ == "__main__":
    main()
"""


hidden1=my_dense_layer(X,n_hidden_1)
hidden2=my_dense_layer(hidden1,n_hidden_2)
hidden3=my_dense_layer(hidden2,n_hidden_3)
outputs=my_dense_layer(hidden3,n_outputs, activation=None)

reconstruction_loss=tf.reduce_mean(tf.square(outputs-X))#MSE

reg_losses=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
							#Graph.Keys.REGULARISATION_LOSSES
							#хронит потери L1, L2 регуляризации
loss=tf.add_n([reconstruction_loss]+reg_losses)

optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)


init = tf.global_variables_initializer()
saver=tf.train.Saver()

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		j=0
		n_batches=len(X_train) // batch_size
		for X_batch, y_batch in shuffle_batch(X_train, y_train, n_batches):
			sess.run(training_op, feed_dict={X: X_batch})
			j=j+1
			if j==200 or j==0:
				loss_train=loss.eval(feed_dict={X: X_batch})
				print("\r{}".format(epoch), "Train MSE:", loss_train)
				j=0
				X_show0=X_batch[8, :].reshape(1,  n_inputs)
				X_show0=X_show0.reshape(28, 28)
				plot_image(X_show0)
				plt.show()
				X_show=outputs.eval(feed_dict={X: X_batch[8, :].reshape(1,  n_inputs)})
				X_show=X_show.reshape(28, 28)
				plot_image(X_show)
				plt.show()


print("finish")

