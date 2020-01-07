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
   
# expected output 
-40.0 degrees in celsius = -40.0 degree in fahrenheit
-10.0 degrees in celsius = 14.0 degree in fahrenheit
0.0 degrees in celsius = 32.0 degree in fahrenheit
8.0 degrees in celsius = 46.0 degree in fahrenheit
15.0 degrees in celsius = 59.0 degree in fahrenheit
22.0 degrees in celsius = 72.0 degree in fahrenheit
38.0 degrees in celsius = 100.0 degree in fahrenheit
# define the exact layers you want your model to train on
10 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([10])

model = tf.keras.Sequential(
    [
     tf.keras.layers.Dense(units=1, input_shapes=[1])
    ]
)
# compile the trained model for loss and gain gauging
model.compile(loss="mean_squared_error",
              optimizer= tf.keras.optimizers.Adam(0.1))

# provide how many times the model should train on 
history= model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("finished Training model")

# output
# finished Training model

# plot the training rate against the data
import matplotlib.pyplot as plt
plt.xlabel('epoch number')
plt.ylabel('Loss magnitude')
plt.plot(history.history['loss'])

print(model.predict([100]))
#outputs
#[[211.30353]] close to exact 212

print("This are the layers variables: {}".format(model.get_weights()))

# prints the labels(data input in arrays form)
# This are the layers variables: [array([[1.8251822]], dtype=float32), array([28.785307], dtype=float32)]



