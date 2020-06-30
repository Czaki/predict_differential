import pickle
import sys

import numpy as np
import tensorflow as tf

from normalizer import Normalizer

tf.config.set_visible_devices([], 'GPU')

def evaluate(model_path: str, input_data: str, normalizer:str, save_path: str):
    with open(normalizer, 'rb') as ff:
        normalizer: Normalizer = pickle.load(ff)
    input_data = normalizer.normalize_input(np.loadtxt(input_data, delimiter=","))
    model = tf.keras.models.load_model(model_path)
    predicted = model.predict(input_data)
    predicted = normalizer.restore_output(predicted)
    np.savetxt(save_path, predicted, delimiter=",")


def main():
    print("begin", sys.argv)
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    print("end")


if __name__ == '__main__':
    main()
