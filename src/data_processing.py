from abc import ABC, abstractmethod
from typing import Type

class ModelAdapter(ABC):
    @abstractmethod
    def load_model(self, model_name: str) -> Type:
        pass


class BaseModel(ABC, ModelAdapter):
    @abstractmethod
    def load_data(self, data_path: str):
        pass

    
class DINOV2(BaseModel):
    def __init__(self, n_nodes, path_manager):
        self.n_nodes = n_nodes

    def load_data(self, data_path: str):
        pass
        

class SAM2(BaseModel):
    def load_data(self, data_path: str):
        pass


class PathManager:
    def __init__(self, config_file_path, output_path, t):
        self.config_file_path
        self.train_dataset_path
        self.output_path
