import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from openclip.src.open_clip.tokenizer import decode
from openclip.src.open_clip.factory import create_model_from_pretrained, create_model_and_transforms, get_tokenizer

from base_model import BaseModel

class OpenCLIP(BaseModel):
    def __init__(
            self, 
            model_name, 
            model_architecture, 
            pretrained=True, 
            pretrained_model=None):
        self.model_name = model_name
        self.model_architecture = model_architecture
        self.pretrained = pretrained
        self.pretrained_model = pretrained_model

    def show_pretrained_models(self):
        pass

    # 이미지 확인용
    def show_image(self, image_source):
        image = Image.open(image_source)
        image = np.array(image.convert("RGB"))

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('on')
        plt.show()
    

    def load_model(self):
        # pretrained model만을 가정하므로 preprocess_train은 반환하지 않음.
        model, _, preprocess = create_model_and_transforms(self.model_architecture, pretrained=self.pretrained_model)
        model.eval()
        tokenizer = get_tokenizer(self.model_architecture)

        return model, tokenizer, preprocess

    def preprocessing(self, image_source, text_source, preprocess, tokenizer):
        image = preprocess(Image.open(image_source).convert('RGB')).unsqueeze(0)
        text = tokenizer(text_source)
        # print(f"preprocessing : image : {image}")
        return image, text
    
    def preprocessing_image(self, image_source, preprocess):
        im = Image.open(image_source).convert('RGB')
        im = preprocess(im).unsqueeze(0)
        return im
    
    def inference(self, model, image, text): 
        # 가공한 이미지, 텍스트를 입력으로 받아 유사도 계산
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image) # 333
            text_features = model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # print(f"image_features : {image_features}")
            # print(f"text_features : {text_features}")
            
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            print("Label probs:", text_probs)
        
        return text_probs
    
    def generate_text(self, model, image_source, preprocess):
        with torch.no_grad(), torch.cuda.amp.autocast():
            image = self.preprocessing_image(image_source, preprocess)
            generated = model.generate(image)
            print(decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
        return generated

def main():
    print("#### Use pretrained model ####")
    IMAGE_PATH = "data/horse.jpg" # 텍스트를 생성할 이미지
    TEXT_SOURCE = ['a cat', 'a dog', 'a horse']

    # model = OpenCLIP(
    #     'openclip', 
    #     model_architecture='ViT-B-32', 
    #     pretrained=True, 
    #     pretrained_model='laion2b_s34b_b79k')

    model = OpenCLIP(
        'openclip', 
        model_architecture='coca_ViT-L-14', 
        pretrained=True, 
        pretrained_model='mscoco_finetuned_laion2B-s13B-b90k')
    
    # input = model.show_image(image_source=IMAGE_PATH)

    loaded_model, tokenizer, preprocess = model.load_model()

    # image, text = model.preprocessing(image_source=IMAGE_PATH, text_source=TEXT_SOURCE, preprocess=preprocess, tokenizer=tokenizer)
    # text_probs = model.inference(loaded_model, image, text)

    model.generate_text(loaded_model, IMAGE_PATH, preprocess)

if __name__ == "__main__":
    main()