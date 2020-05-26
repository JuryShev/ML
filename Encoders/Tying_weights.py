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

def plot_image(image0, image1):
    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plt.imshow(image0, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(image1, cmap="gray", interpolation="nearest")
    plt.show()
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

activation=tf.nn.elu
X=tf.placeholder(tf.float32, [None, n_inputs])
he_init=tf.contrib.layers.variance_scaling_initializer() #initializer Xavier for wight in range sqrt(6/n_outputs+n_inputs)
l2_regularizer=tf.contrib.layers.l2_regularizer(l2_reg)

w1_init=he_init([n_inputs, n_hidden_1])
w2_init=he_init([n_hidden_1, n_hidden_2])

w1=tf.Variable(w1_init, dtype=tf.float32, name="w1")
w2=tf.Variable(w2_init, dtype=tf.float32, name="w2")
w3=tf.transpose(w2, name="w3")
w4=tf.transpose(w1, name="w4")

bias_1=tf.Variable(tf.zeros(n_hidden_1), name="bias_1")
bias_2=tf.Variable(tf.zeros(n_hidden_2), name="bias_2")
bias_3=tf.Variable(tf.zeros(n_hidden_3), name="bias_3")
bias_4=tf.Variable(tf.zeros(n_outputs), name="bias_4")

hidden1=activation(tf.matmul(X, w1)+bias_1)
hidden2=activation(tf.matmul(hidden1, w2)+bias_2)
hidden3=activation(tf.matmul(hidden2, w3)+bias_3)
outputs=tf.matmul(hidden3, w4)+bias_4

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
			if j==800 or j==0:
				loss_train=loss.eval(feed_dict={X: X_batch})
				print("\r{}".format(epoch), "Train MSE:", loss_train)
				j=0
				X_show0=X_batch[8, :].reshape(1,  n_inputs)
				X_show0=X_show0.reshape(28, 28)

				X_show=outputs.eval(feed_dict={X: X_batch[8, :].reshape(1,  n_inputs)})
				X_show=X_show.reshape(28, 28)
				plot_image(X_show0, X_show)



print("finish")