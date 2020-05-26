import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import gauss
n_steps=20
n_steps2=100
n_inputs=1
n_neurons=100
n_outputs = 1

n_epochs = 500
batch_size = 1

t_min, t_max, t_max2 = 0, 30, 31
resolution = 0.1
resolution2 = 0.01



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
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, 1)
"""_____________________________________________________________________________"""

t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution)) # break down a series of values.
                                                                 # int((t_max - t_min) / resolution) number of values
t2 = np.linspace(t_min, t_max2, int((t_max2 - t_min) / resolution))
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1) #Series of instance for the assessment function
t_instance2 = np.linspace(15, 16 , n_steps + 1)

reset_graph()


X = tf.placeholder(tf.float32, shape=(None, n_steps, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None, n_steps, n_inputs), name="y")

cell=tf.keras.layers.SimpleRNNCell(n_neurons) #manufacture n_neurons


rnn = tf.keras.layers.RNN(
    cell,
    return_sequences=True,
    return_state=True) #Creat RNN

# Full series           final series
whole_sequence_output, final_state = rnn(X)

stacked_rnn_outputs = tf.reshape(whole_sequence_output, [-1, n_neurons])# rehape [batch_size*steps, n_neurons ] for fully
                                                                        #  connected appelement
stacked_outputs=tf.layers.dense(stacked_rnn_outputs, n_outputs, name="stacked_outputs") # fully connected appelement
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs]) #rehape [batch_size, steps, n_neurons ] for RNN

init = tf.global_variables_initializer()
saver=tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./my_time_series_model")

            #############draw in the 0###################
    sequence1 = [0. for i in range(n_steps)] # select series in the function predict
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence1[-n_steps:]).reshape(1, n_steps, 1)# reshape for input-X
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence1.append(y_pred[0, -1, 0])# Add data in the end
    ###################################################################################

			#############draw in the 15(instance2)###################
#i * resolution + t_min + (t_max - t_min / 3)
    sequence2 = [time_series(i) for i in t_instance2]
    series=len(sequence2)-n_steps
    plt.plot(t[:len(sequence2)], sequence2, "b-")
    plt.show()
    for iteration in range(len(t) - len(sequence2)):
        X_batch = np.array(sequence2[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence2.append(y_pred[0, -1, 0])
	###################################################################################
	
plt.figure(figsize=(11, 4))
plt.subplot(121)
plt.plot(t, sequence1, "b-")
plt.plot(t[:n_steps], sequence1[:n_steps], "b-", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.plot(t, sequence2, "b-")
plt.plot(t[series:series+n_steps], sequence2[series:series+n_steps],  "r-", linewidth=3)
plt.xlabel("Time")
plt.show()
print("finish")

