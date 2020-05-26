import tensorflow as tf
import numpy as np
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt



n_inputs = 28 * 28
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20  # codings
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs

n_epochs=50
batch_size=150
learning_rate=0.0001
eps = 1e-10

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def shuffle_batch_encoder(X, n_batches):
    rnd_idx = np.random.permutation(len(X))
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch = X[batch_idx]
        yield X_batch

def plot_image(image0, image1):
    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    image0 = image0.reshape(28, 28)
    image1 = image0.reshape(28, 28)
    plt.imshow(image0, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(image1, cmap="gray", interpolation="nearest")
    plt.show()

reset_graph()

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
X=tf.placeholder(tf.float32, [None, n_inputs])

he_init=tf.contrib.layers.variance_scaling_initializer() #initializer Xavier for wight in range sqrt(6/n_outputs+n_inputs)
#l2_regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
my_dense_layer=partial(tf.layers.dense, activation=tf.nn.elu,
					   kernel_initializer=he_init,
					   )



hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3_mean = my_dense_layer(hidden2, n_hidden3, activation=None)
hidden3_sigma = my_dense_layer(hidden2, n_hidden3, activation=None)
noise = tf.random_normal(tf.shape(hidden3_sigma), dtype=tf.float32)
hidden3 = hidden3_mean + hidden3_sigma * noise
hidden4 = my_dense_layer(hidden3, n_hidden4)
hidden5 = my_dense_layer(hidden4, n_hidden5)
logits = my_dense_layer(hidden5, n_outputs, activation=None)
outputs = tf.sigmoid(logits)



xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X,logits=logits)

reconstruction_loss = tf.reduce_sum(xentropy)
latent_loss = 0.5 * tf.reduce_sum(
    tf.square(hidden3_sigma) + tf.square(hidden3_mean)
    - 1 - tf.log(eps + tf.square(hidden3_sigma)))

loss = reconstruction_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver=tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        j=0
        n_batches=len(X_train) // batch_size
        for X_batch in shuffle_batch_encoder(X_train,  n_batches):
            out_0=sess.run(training_op, feed_dict={X: X_batch})
            out_sigma=hidden3_sigma.eval(feed_dict={X: X_batch})
            j=j+1
            if j == n_batches:
                loss_train = latent_loss.eval(feed_dict={X: X_batch})
                print("\r{}".format(epoch), "Train MSE:", loss_train)
                j = 0
    codings_rnd = np.random.normal(size=[2, n_hidden3])

    outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})

plot_image(outputs_val[0], outputs_val[1])
print("finish")



