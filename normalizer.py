from abc import ABC, abstractmethod

import numpy as np

class Normalizer(ABC):
    @abstractmethod
    def normalize_input(self, data: np.ndarray) -> np.ndarray:
        raise

    @abstractmethod
    def normalize_output(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def restore_output(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def restore_input(self, data: np.ndarray) -> np.ndarray:
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

    def restore_output(self, data: np.ndarray) -> np.ndarray:
        return data * np.subtract(self.output_max, self.output_min) + self.output_min

    def restore_input(self, data: np.ndarray) -> np.ndarray:
        return data * np.subtract(self.input_max, self.input_min) + self.input_min


class StdInputNormalizer(Normalizer):
    def __init__(self, input_df: np.ndarray):
        self.input_mean = list(input_df.mean(axis=0))
        self.input_std = list(input_df.std(axis=0))

    def normalize_input(self, data: np.ndarray):
        return (data - self.input_mean) / self.input_std

    def restore_input(self, data: np.ndarray) -> np.ndarray:
        return data * self.input_std + self.input_mean


class StdNormalizer(StdInputNormalizer):
    def __init__(self, input_df: np.ndarray, output_df: np.ndarray):
        super().__init__(input_df)
        self.output_mean = list(output_df.mean(axis=0))
        self.output_std = list(output_df.std(axis=0))

    def normalize_output(self, data: np.ndarray):
        return (data - self.output_mean) / self.output_std

    def restore_output(self, data: np.ndarray) -> np.ndarray:
        return data * self.output_std + self.output_mean

class NoNormalizer(Normalizer):
    def __init__(self):
        pass 
    
    def normalize_input(self, data: np.ndarray) -> np.ndarray:
        return data 

    def normalize_output(self, data: np.ndarray) -> np.ndarray:
        return data 

    def restore_output(self, data: np.ndarray) -> np.ndarray:
        return data

    def restore_input(self, data: np.ndarray) -> np.ndarray:
        return data

class StdInputNormalizerOnly(StdInputNormalizer, NoNormalizer):
    pass

class IncreaseNormalizer(Normalizer):
    def __init__(self):
        pass

    def normalize_input(self, data: np.ndarray) -> np.ndarray:
        return data

    def normalize_output(self, data: np.ndarray) -> np.ndarray:
        res = np.copy(data)
        res[:, 1:] -= data[:, :-1]
        return res

    def restore_output(self, data: np.ndarray) -> np.ndarray:
        return np.cumsum(data, axis=1)

    def restore_input(self, data: np.ndarray) -> np.ndarray:
        return data


class IncreaseNormNormalizer(Normalizer):
    def __init__(self, input_df: np.ndarray, output_df: np.ndarray):
        self._normalizer = StdNormalizer(input_df, self._normalize_output(output_df))

    def normalize_input(self, data: np.ndarray) -> np.ndarray:
        return data

    def _normalize_output(self, data: np.ndarray):
        res = np.copy(data)
        res[:, 1:] -= data[:, :-1]
        return res

    def normalize_output(self, data: np.ndarray) -> np.ndarray:
        return self._normalizer.normalize_output(self._normalize_output(data))

    def restore_output(self, data: np.ndarray) -> np.ndarray:
        return self._normalizer.restore_output(np.cumsum(data, axis=1))

    def restore_input(self, data: np.ndarray) -> np.ndarray:
        return data