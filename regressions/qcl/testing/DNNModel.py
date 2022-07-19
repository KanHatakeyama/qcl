from pickletools import optimize
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import Adam

"""
MLP class with Keras backend
"""


def prepare_dnn_model(activation="tanh", hidden_dim=16, layers=2):
    dnn = Sequential()
    dnn.add(Dense(1,  input_shape=[1]))
    for i in range(layers):
        dnn.add(Dense(hidden_dim, activation=activation))
    dnn.add(Dense(1))

    dnn.compile(loss='mse',
                # optimizer="adam",
                optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999),
                metrics=['mse'])

    return dnn


class DNNModel:
    def __init__(self, activation="relu", hidden_dim=16, layers=2):
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.layers = layers

    def fit(self, x, y, epochs=1000, batch_size=30, verbose=0):
        self.dnn = prepare_dnn_model(
            self.activation, self.hidden_dim, layers=self.layers)
        self.dnn.fit(x, y, epochs=epochs,
                     batch_size=batch_size, verbose=verbose)

        self.hidden_model = Model(
            inputs=self.dnn.input, outputs=self.dnn.get_layer(self.dnn.layers[-2].name).output)

    def predict(self, x):
        return self.dnn.predict(x).reshape(-1)

    def hidden_predict(self, x):
        return self.hidden_model.predict(x)
