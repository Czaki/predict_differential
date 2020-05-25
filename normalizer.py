from abc import ABC, abstractmethod

import numpy as np

class Normalizer(ABC):
    @abstractmethod
    def normalize_input(self, data: np.ndarray) -> np.ndarray:
        pass

    def normalize_output(self, data: np.ndarray) -> np.ndarray:
        pass


class UniformNormalizer(Normalizer):
    def __init__(self, input_df: np.ndarray, output_df: np.ndarray):
        self.input_min = list(input_df.min(axis=0))
        self.output_min = list(output_df.min(axis=0))
        self.input_max = list(input_df.max(axis=0))
        self.output_max = list(output_df.max(axis=0))

    def normalize_input(self, data: np.ndarray):
        return (data - self.input_min) / np.subtract(self.input_max, self.input_min)

    def normalize_output(self, data: np.ndarray):
        return (data - self.output_min) / np.subtract(self.output_max, self.output_min)


class StdNormalizer(Normalizer):
    def __init__(self, input_df: np.ndarray, output_df: np.ndarray):
        self.input_mean = list(input_df.mean(axis=0))
        self.output_mean = list(output_df.mean(axis=0))
        self.input_std = list(input_df.std(axis=0))
        self.output_std = list(output_df.std(axis=0))

    def normalize_input(self, data: np.ndarray):
        return (data - self.input_mean) / self.input_std

    def normalize_output(self, data: np.ndarray):
        return (data - self.output_mean) / self.output_std