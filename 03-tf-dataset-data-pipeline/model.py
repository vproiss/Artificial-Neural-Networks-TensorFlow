import tensorflow as tf

class Model(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = tf.keras.layers.Dense(units=10,
                                                    activation=tf.keras.activations.sigmoid)
        self.hidden_layer_2 = tf.keras.layers.Dense(units=10,
                                                    activation=tf.keras.activations.sigmoid)
        self.output_layer = tf.keras.layers.Dense(units=1,
                                                  activation=tf.keras.activations.sigmoid)

    def call(self, inputs):
        x = self.hidden_layer_1(inputs)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)
        return x
