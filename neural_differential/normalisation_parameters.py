import argparse
from typing import List, NamedTuple, Union

import numpy as np


class NormalisationParameters(NamedTuple):
    mean: np.ndarray
    std: np.ndarray


class DataNormalisation(NamedTuple):
    inputs: NormalisationParameters
    outputs: NormalisationParameters


def prepare_data(input_files: List[str], output_files: List[str], save_file):
    inputs = np.concatenate([np.loadtxt(file_name, delimiter=",") for file_name in input_files])
    outputs = np.concatenate([np.loadtxt(file_name, delimiter=",") for file_name in output_files])
    input_std = np.std(inputs, axis=0)
    input_std[input_std == 0] = 1
    output_std = np.std(outputs, axis=0)
    output_std[output_std == 0] = 1
    res = {
        "input_mean": np.mean(inputs, axis=0),
        "input_std": input_std,
        "output_mean": np.mean(outputs, axis=0),
        "output_std": output_std,
    }
    np.savez(save_file, **res)


def load_data(save_file: str) -> DataNormalisation:
    res = np.load(save_file)
    return DataNormalisation(
        NormalisationParameters(res["input_mean"], res["input_std"]),
        NormalisationParameters(res["output_mean"], res["output_std"]),
    )


def normalize_input(parameters: Union[DataNormalisation, NormalisationParameters], data: np.ndarray) -> np.ndarray:
    if isinstance(parameters, DataNormalisation):
        parameters = parameters.inputs
    return (data - parameters.mean) / parameters.std


def normalize_output(parameters: Union[DataNormalisation, NormalisationParameters], data: np.ndarray) -> np.ndarray:
    if isinstance(parameters, DataNormalisation):
        parameters = parameters.outputs
    return (data - parameters.mean) / parameters.std


def main():
    parser = argparse.ArgumentParser("Prepare data")
    parser.add_argument(
        "-i,--input_files", nargs="+", help="list of input csv files", required=True, dest="inputs",
    )
    parser.add_argument(
        "-o,--output_files", nargs="+", help="list of output csv files", required=True, dest="outputs",
    )
    parser.add_argument(
        "-s,--save", nargs=1, help="files in which save normalization parameters", required=True, dest="save",
    )

    args = parser.parse_args()
    prepare_data(args.inputs, args.outputs, args.save[0])


if __name__ == "__main__":
    main()
