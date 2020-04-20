import argparse
from typing import List

import tensorflow as tf


def multiple_features(input_size, output_size, layers_size: List[int], activation="relu"):
    inputs = tf.keras.layers.Input(input_size)
    res_layers = []
    for i in range(output_size):
        previous_layer = inputs
        for val in layers_size:
            previous_layer = tf.keras.layers.Dense(val, activation=activation)(previous_layer)
        res_layers.append(tf.keras.layers.Dense(1)(previous_layer))
    outputs = tf.keras.layers.concatenate(res_layers)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss="mse", optimizer=optimizer, metrics=["mae", "mse"])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("layers", nargs="+", type=int)
    parser.add_argument("save_path", type=str)
    args = parser.parse_args()
    model = multiple_features(7, 200, args.layers)
    model.save(args.save_path)
