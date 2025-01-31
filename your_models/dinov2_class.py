from base_model import BaseModel
import torch
from PIL import Image
import os
import torchvision.transforms as T
import numpy as np
from typing import List

from dinov2.models.vision_transformer import vit_base, vit_small, vit_large, vit_giant2, DinoVisionTransformer


script_path = os.path.dirname(os.path.realpath(__file__))


class DINOV2(BaseModel):
    def __init__(self):
        # torch가 사용할 device 설정
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _parse_model_scale(self):
        return self.model_name[3]


    def load_model(self, model_name, patch_size=14, img_size=518, use_pretrained=True, use_register=False):
        # Register 모델 사용 여부
        self.model_name = model_name
        self.pretrained = use_pretrained

        use_register = use_register


        # 예를 들어, self.mode_name이 "vitb14"라면 model_scale은 "b"임. 
        model_scale = self._parse_model_scale()
        models = {
            "b": vit_base,
            "s": vit_small,
            "l": vit_large,
            "g": vit_giant2,
        }


        # 사용자가 원하는 모델 스케일에 해당하는 모델 아키텍쳐를 self.model에 할당함.
        try:
            self.model = models[model_scale](
                patch_size=patch_size,
                img_size=img_size,
                init_values=1.0,
                block_chunks=0,
            )

        except KeyError as e:
            print(f"[KeyError {e}] Please check supproted models: vitb14, vits14, vitl14, vitg14")
            exit(-1)
            

        # Pretrained 된 .pth 파일 가져와서 self.model에 적용.
        try:
            # register가 사용된 모델을 사용할지 여부 결정
            if use_register:
                checkpoint_file_name = f"dinov2_{self.model_name}_reg4_pretrain.pth"
            checkpoint_file_name = f"dinov2_{self.model_name}_pretrain.pth"

            checkpoint_file_path = os.path.join(script_path, f"../checkpoints/dinov2/backbones/{checkpoint_file_name}")
            print(checkpoint_file_path)
            checkpoint = torch.load(checkpoint_file_path)

            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
        
        except FileNotFoundError as e:
            print(f"[{e.strerror}] '{checkpoint_file_name}' is not found. Please check the supported models from README.md.")
            exit(-1)

        except RuntimeError as e:
            print(f"Failed to load the model due to a runtime error: {e}")
            exit(-1)


        # 모델을 평가 모드로 전환
        self.model.eval()


    def compute_embeddings(self, images) -> List[dict]:
        all_embeddings = []
        with torch.no_grad():
            for image in images:
                image_embedding = self.model(image.to(self.device))
                embeddings = np.array(image_embedding[0].detach().cpu().numpy()).reshape(1, -1)
                all_embeddings.append(embeddings)

        return all_embeddings


    def _load_data(self, image_path) -> List[Image.Image]:
        all_images = []

        image_dir = os.fsencode(image_path)

        # 테스트를 위해서 10개의 이미지만 로드함. 추후 변경
        for file in os.listdir(image_dir)[:10]:
            filename = os.fsdecode(file)
            if filename.endswith(".jpg") or filename.endswith(".JPEG"): 
                image = Image.open(os.path.join(image_path, filename))
                all_images.append(image)
            else:
                continue

        return all_images

        
    # dinov2 모델에 호환되는 input 형태로 이미지 가공
    def preprocess_input_data(self, image_path):
        all_transfomred_images = []

        # 이미지 데이터 로드
        all_images = self._load_data(image_path)

        # torchvision.transforms 모듈의 Compose 객체
        transform_image = T.Compose([T.ToTensor(), T.Resize(244, antialias=True), T.CenterCrop(224), T.Normalize([0.5], [0.5])])

        for image in all_images:
            transformed_image = transform_image(image)[:3].unsqueeze(0)
            all_transfomred_images.append(transformed_image)

        return all_transfomred_images

if __name__ == "__main__":
    # 사전 훈련된 모델 사용하기 (Local)
    IMAGE_PATH = os.path.join(script_path, "../data/flickr30k/Images")
    dinov2 = DINOV2()
    dinov2.load_model('vits14')
    images = dinov2.preprocess_input_data(IMAGE_PATH)
    # for i in images:
    #     print(i.shape)
    embedding_results = dinov2.compute_embeddings(images)
    print(embedding_results[0].shape)
    print(embedding_results)