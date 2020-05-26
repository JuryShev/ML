import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import gauss

n_steps=20
n_inputs=1
n_neurons=100
n_outputs = 1

n_epochs = 1500
batch_size = 50

t_min, t_max = 0, 30
resolution = 0.1


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

"""_______________Function predict_____________________________________________"""
def time_series(t):
    return np.sin(t)+gauss(0.0, 0.1)#t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)
"""_____________________________________________________________________________"""


reset_graph()

t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))# break down a series of values.
                                                                 # int((t_max - t_min) / resolution) number of values
																 
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)#Series of instance for the assessment function

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.title("A time series (generated)", fontsize=14)
plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label="A training instance")
plt.legend(loc="lower left", fontsize=14)
plt.axis([0, 30, -17, 13])
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.title("A training instance", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.legend(loc="upper left")
plt.xlabel("Time")



#plt.show()

X = tf.placeholder(tf.float32, shape=(None, n_steps, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None, n_steps, n_inputs), name="y")




cell=tf.keras.layers.SimpleRNNCell(n_neurons)#manufacture n_neurons

rnn = tf.keras.layers.RNN(
    cell,
    return_sequences=True,
    return_state=True)#Creat RNN
	
# Full series           final series
whole_sequence_output, final_state = rnn(X)
stacked_rnn_outputs = tf.reshape(whole_sequence_output, [-1, n_neurons])# rehape [batch_size*steps, n_neurons ] for fully
                                                                        #  connected appelement
																		
stacked_outputs=tf.layers.dense(stacked_rnn_outputs, n_outputs, name="stacked_outputs")# fully connected appelement
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])#rehape [batch_size, steps, n_neurons ] for RNN

with tf.name_scope("train"):
    learning_rate = 0.001
    loss = tf.reduce_mean(tf.square(outputs-y), name="loss")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver=tf.train.Saver()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):

        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(epoch, "\tMSE:", mse)

    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})

    saver.save(sess, "./my_time_series_model") # not shown in the book
print("finish")


