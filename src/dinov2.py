from base_model import BaseModel
import torch
import requests
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification
from PIL import Image

class DINOV2(BaseModel):
    def __init__(self, model_name, pretrained=True):
        self.model_name = model_name
        self.pretrained = pretrained

    def process_image(self, image_url):
        # image_url로부터 불러온 사진을 AutoImageProcessor가 전처리해서 모델에 적합한 input 형태로 변환함.
        if self.pretrained:
            try:
                image = Image.open(requests.get(image_url, stream=True).raw)
            except Exception as e:
                print("Error: Failed to process the image. Please try again.")
                exit(0)
        processor = AutoImageProcessor.from_pretrained(self.model_name)
        inputs = processor(images=image, return_tensors="pt")

        return inputs
    

    def classify_image(self, inputs):
        # AutoModelForImageClassification을 사용하여 모델을 불러옴
        model = AutoModelForImageClassification.from_pretrained(self.model_name)

        # 모델을 통해 예측을 수행
        outputs = model(**inputs)
        logits = outputs.logits  # logits는 모델의 최종 출력 (분류 점수)
        return logits
    

    def trace_model(self, inputs):
        self.model = AutoModel.from_pretrained(self.model_name)
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]

        self.model.config.return_dict = False

        with torch.no_grad():
            traced_model = torch.jit.trace(self.model, [inputs.pixel_values])
            traced_outputs =traced_model(inputs.pixel_values)

        traced_result = (last_hidden_states - traced_outputs[0]).abs().max()
        
        return traced_result


if __name__ == "__main__":
    # 사전 훈련된 허깅페이스 모델 사용하기 (Online)
    IMAGE_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    model = DINOV2('facebook/dinov2-base', pretrained=True)
    inputs = model.process_image(image_url=IMAGE_URL)
    logits = model.classify_image(inputs)
    print(f"logits: {logits}\n")
    traced_result = model.trace_model(inputs)
    if traced_result:
        print(f"successfully used pretrained {model.model_name}.\ntraced_result: {traced_result}")