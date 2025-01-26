import torch
import faiss
import numpy as np
from PIL import Image
import csv
from dinov2_class import preprocess_input_data, DINOV2


class FaissConstructor:
    def __init__(self, model):
        self.model = model
        self.index = faiss.IndexFlatL2(384)


    def add_vector_to_index(self, embedding):
        #convert embedding to numpy
        vector = embedding.detach().cpu().numpy()

        print("vector: ", vector)
        #Convert to float32 numpy
        vector = np.float32(vector)

        #Normalize vector: important to avoid wrong results when searching
        faiss.normalize_L2(vector)

        #Add to index
        self.index.add(vector)


    def extract_embeddings(self, images):
        for image_path in images:
            image = Image.open(image_path).convert('RGB')
            with torch.no_grad():
                inputs = preprocess_input_data(image_path)
                features = self.model.compute_embeddings(inputs)
                print(features[0].mean())
            self.add_vector_to_index(embedding=features.mean(dim=1))

    def write_index(self, vector_index):
        faiss.write_index(self.index, vector_index)



if __name__ == "__main__":
    dinov2 = DINOV2()
    dinov2.load_model('vits14')
    fc = FaissConstructor(dinov2)
    IMAGE_PATH = ['/home/baesik/24Winter_OOP_AI_Agent/data/cat10.jpg']
    fc.extract_embeddings(IMAGE_PATH)
    fc.write_index("vector.index")