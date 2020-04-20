import argparse
import os
import glob

import numpy as np
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow as tf


from neural_differential.normalisation_parameters import load_data, normalize_input, normalize_output


def run_model(input_data, output_data, model_chk, epochs):
    model = tf.keras.models.load_model(model_chk)
    history = model.fit(
        input_data,
        output_data[:, : model.output_shape[1]],
        epochs=epochs,
        validation_split=0.2,
        verbose=0,
        callbacks=[tfdocs.modeling.EpochDots()],
        batch_size=2 * 14,
    )
    model.save(model_chk)


def main():
    parser = argparse.ArgumentParser("Simple model run")
    parser.add_argument("data_dir")
    parser.add_argument("norm_parameters")
    parser.add_argument("model")
    parser.add_argument("--epochs", type=int, default=1000, dest="epochs")

    args = parser.parse_args()
    input_files = sorted(glob.glob(os.path.join(args.data_dir, "input*csv")))
    output_files = sorted(glob.glob(os.path.join(args.data_dir, "output*csv")))
    norm_param = load_data(args.norm_parameters)
    inputs = np.concatenate([np.loadtxt(file_name, delimiter=",") for file_name in input_files])
    outputs = np.concatenate([np.loadtxt(file_name, delimiter=",") for file_name in output_files])
    inputs = normalize_input(norm_param, inputs)
    outputs = normalize_output(norm_param, outputs)
    run_model(inputs, outputs[:, 1:], args.model, args.epochs)


if __name__ == "__main__":
    main()
