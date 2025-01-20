from abc import ABC, abstractmethod

class BaseModel:
    model_name = None
    pretrained = False
    model = None

    @abstractmethod
    def process_input(self, image_source):
        pass

    @abstractmethod
    def do_train(self, inputs):
        pass

    @abstractmethod
    def get_eval(self, metric):
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