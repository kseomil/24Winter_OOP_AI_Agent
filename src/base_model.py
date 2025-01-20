from abc import ABC, abstractmethod

class BaseModel:
    model_name = None
    pretrained = False

    @abstractmethod
    def set_pretrained(self, pretrained: bool):  
        pass      

    # @abstractmethod
    # def load_model(self):
    #     pass

    # @abstractmethod
    # def load_data(self):
    #     pass

    # @abstractmethod
    # def train(self, input_data):
    #     pass

    # @abstractmethod
    # def evaluate(self, output_data):
    #     pass