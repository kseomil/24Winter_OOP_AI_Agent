import torch
import faiss
import numpy as np
from PIL import Image
import csv
import os
from dinov2_class import preprocess_input_data, DINOV2


script_path = os.path.dirname(os.path.realpath(__file__))


class FaissConstructor:
    def __init__(self, model):
        self.model = model
        self.index = faiss.IndexFlatL2(384)


    def add_vector_to_index(self, all_embeddings: list):
        """
            embeddings vector를 정규화해서 self.index에 추가함.
        """
        for embeddings in all_embeddings:
            embeddings = np.float32(embeddings)
            faiss.normalize_L2(embeddings)
            #Add to index
            self.index.add(embeddings)


    def write_index(self, vector_index):
        faiss.write_index(self.index, vector_index)
        print(f"Successfully created {vector_index}")


    def search_k_similar_images(self, vector_index, input_image, k=1):
        # index 파일 불러오기
        index = faiss.read_index(vector_index)

        # OpenClip에서 추출된 이미지를 dinov2 모델에 입력 가능한 형태로 변환하여 임베딩 계산
        input_image_embeddings = self.model.compute_embeddings(preprocess_input_data(input_image))[0]
        print(input_image_embeddings)

        # FAISS 검색 수행
        distances, indices = index.search(input_image_embeddings, k)

        # 결과 출력
        print("in FaissConstructor.search_k_similar_images: ")
        print("distance: ", distances[0][0], " indices: ", indices[0][0])

        return indices
    

if __name__ == "__main__":
    IMAGE_PATH = IMAGE_PATH = os.path.join(script_path, "../data/flickr30k/Images")
    INEDEX_PATH = os.path.join(script_path, "../vector.index")

    dinov2 = DINOV2()
    dinov2.load_model('vits14')
    images = preprocess_input_data(IMAGE_PATH)
    embedding_results = dinov2.compute_embeddings(images)

    fc = FaissConstructor(dinov2)
    fc.add_vector_to_index(embedding_results)
    fc.write_index("vector.index")

    target = os.path.join(script_path, "../data/val/")

    image_index = fc.search_k_similar_images(INEDEX_PATH, input_image=target)
    print(image_index[0][0])