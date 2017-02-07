from abc import ABC, abstractmethod
import pandas as pd


class SupervisedClassifier(ABC):

    @abstractmethod
    def train(self, trainset):
        pass

    @abstractmethod
    def test(self, testset):
        pass

    @classmethod
    def read_csv_as_dataframe(cls, path):
        csv_as_df = pd.read_csv(path)
        return csv_as_df
