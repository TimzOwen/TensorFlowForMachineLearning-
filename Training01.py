import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.ERROR)


def dense_network(C):
    f_height = C * 1.82 + 32
    return f_height


print(dense_network(0))
print(dense_network(10))

using supervised Learning to train the data
given set of inputs and outputs and figures out the Dense pattern (Algorithm)
create arrays for input and output data

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# print output using fro loop
for i, c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees fahrenheit".format(c,fahrenheit_a[i]))
   


































