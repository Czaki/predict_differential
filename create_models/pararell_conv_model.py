import argparse
from typing import List

import tensorflow as tf


def multiple_features(input_size, output_size, layers_size: List[int], kernel_size, activation="relu"):
    inputs = tf.keras.layers.Input(input_size)
    res_layers = []
    for i in range(output_size):
        previous_layer = inputs
        for val in layers_size:
            previous_layer = tf.keras.layers.Dense(val, activation=activation)(previous_layer)
        res_layers.append(tf.keras.layers.Dense(1)(previous_layer))
    concat = tf.keras.layers.concatenate(res_layers)
    reshape = tf.keras.layers.Reshape((output_size, 1))(concat)
    if kernel_size % 2 == 1:
        padd = (kernel_size - 1) // 2
    else:
        padd = (kernel_size - 1) // 2
        padd = padd, padd + 1
    padding = tf.keras.layers.ZeroPadding1D(padd)(reshape)
    conv = tf.keras.layers.Conv1D(1, kernel_size)(padding)
    outputs = tf.keras.layers.Reshape((output_size,))(conv)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss="mse", optimizer=optimizer, metrics=["mae", "mse"])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("layers", nargs="+", type=int)
    parser.add_argument("save_path", type=str)
    parser.add_argument("--conv_size", type=int, default=3, dest="kernel")
    args = parser.parse_args()
    model = multiple_features(7, 10, args.layers, args.kernel)
    model.save(args.save_path)
    tf.keras.utils.plot_model(model, show_shapes=True, to_file="model.png")
