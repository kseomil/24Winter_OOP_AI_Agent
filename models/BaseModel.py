from abc import ABC, abstractmethod

class BaseModel:
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def train(self, input_data):
        pass

    @abstractmethod
    def evaluate(self, output_data):
        pass