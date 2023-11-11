"""
This is the implementation of model proposed in the paper "Expanding Analytical Capabilities in Intrusion Detection
through Ensemble-Based Multi-Label Classification" submitted in Computers & Security, Elsevier.
Copyright (C) 2023 Ehsan Hallaji
All rights reserved.
"""

import tensorflow as tf
import utils


class emlnn:
    def __init__(self, input_size, num_classes,
                 batch_size=64,
                 learning_rate=0.001,
                 hidden_layer_sizes=(256, 256, 128, 64),
                 dropout_ratio=0.4,
                 epochs=100,
                 verbose=1,
                 metrics=['accuracy'],
                 label_output=True):

        if isinstance(metrics, str):
            metrics = [metrics]  # Convert single string to a list
        elif not isinstance(metrics, list):
            raise ValueError("The 'metrics' argument should be a string or a list of strings.")

        for value in [batch_size, learning_rate, dropout_ratio, epochs]:
            if not isinstance(value, (int, float)):
                raise ValueError(f"The value '{value}' should be numeric.")

        if not (0 <= learning_rate <= 1) or not (0 <= dropout_ratio <= 1):
            raise ValueError("Learning rate and dropout ratio should be between 0 and 1.")

        if not isinstance(batch_size, int) or not (batch_size >= 1):
            raise ValueError("Batch size should be an integer equal to or larger than 1.")

        if not isinstance(epochs, int) or not (epochs >= 1):
            raise ValueError("Epochs should be an integer equal to or larger than 1.")

        if not isinstance(verbose, int) or not (0 <= verbose <= 2):
            raise ValueError("Verbose should be an integer between 0 and 2.")

        if not isinstance(label_output, bool):
            raise ValueError("The 'label_output' argument should be a boolean.")

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_ratio = dropout_ratio
        self.epochs = epochs
        self.verbose = verbose
        self.metrics = metrics
        self.label_output = label_output

        self.ensemble = self.create_ensemble(input_size, num_classes)

    def create_ensemble(self, input_size, num_classes):
        ensemble = []
        for i in range(len(num_classes)):
            model = tf.keras.Sequential()

            # Add an initial input layer
            model.add(tf.keras.layers.InputLayer(input_shape=(input_size,)))

            for layer_size in self.hidden_layer_sizes:
                model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
                model.add(tf.keras.layers.Dropout(self.dropout_ratio))

            model.add(tf.keras.layers.Dense(num_classes[i], activation='softmax' if num_classes[i] > 2 else 'sigmoid'))

            if num_classes[i] == 2:
                loss_function = tf.keras.losses.BinaryCrossentropy()
            else:
                loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                          loss=loss_function,
                          metrics=self.metrics)
            ensemble.append(model)
        return ensemble

    def train(self, X_train, train_label_sets):
        for i in range(len(self.ensemble)):
            if i > 0:
                num_layers = len(self.ensemble[i].layers) - 1
                for j in range(num_layers):  # Transfer weights from the previous model, except for the output layer
                    self.ensemble[i].layers[j].set_weights(self.ensemble[i - 1].layers[j].get_weights())
            self.ensemble[i].fit(X_train, train_label_sets[i],
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 verbose=self.verbose)

    def predict(self, x_test):
        predictions = []
        for model in self.ensemble:
            predicted_probabilities = model.predict(x_test)
            if self.label_output:
                predictions.append(utils.convert_to_labels(predicted_probabilities))
            else:
                predictions.append(predicted_probabilities)
        return predictions
