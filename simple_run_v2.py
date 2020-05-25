import argparse
import os
import glob
import pickle

import numpy as np
import tensorflow as tf

from normalizer import Normalizer


def run_model(input_data, output_data, model_chk, epochs):
    print("#"*50)
    print("input size :", input_data.shape)
    print("output size :", output_data.shape)
    print("#"*50)
    model = tf.keras.models.load_model(model_chk)
    history = model.fit(
        input_data,
        output_data[:, model.output_shape[1]],
        epochs=epochs,
        validation_split=0.2,
        verbose=0,
        callbacks=[EpochDots()],
        batch_size=2 ** 14,
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
    with open(args.norm_parameters, 'rb') as ff:
        norm_param: Normalizer = pickle.load(ff)
    inputs = np.concatenate([np.loadtxt(file_name, delimiter=",") for file_name in input_files])
    outputs = np.concatenate([np.loadtxt(file_name, delimiter=",") for file_name in output_files])
    inputs = norm_param.normalize_input(inputs)
    outputs = norm_param.normalize_output(outputs)
    run_model(inputs, outputs, args.model, args.epochs)


class EpochDots(tf.keras.callbacks.Callback):
  """A simple callback that prints a "." every epoch, with occasional reports.

  Args:
    report_every: How many epochs between full reports
    dot_every: How many epochs between dots.
  """

  def __init__(self, report_every=100, dot_every=1):
    self.report_every = report_every
    self.dot_every = dot_every

  def on_epoch_end(self, epoch, logs):
    if epoch % self.report_every == 0:
      print()
      print('Epoch: {:d}, '.format(epoch), end='')
      for name, value in sorted(logs.items()):
        print('{}:{:0.4f}'.format(name, value), end=',  ')
      print()

    if epoch % self.dot_every == 0:
      print('.', end='')



if __name__ == "__main__":
    main()
