from base_model import BaseModel
import torch
import requests
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

class DINOV2(BaseModel):
    def __init__(self, model_name, pretrained=True):
        self.model_name = model_name
        self.pretrained = pretrained

    def process_image(self, image_url):
        try:
            image = Image.open(requests.get(image_url, stream=True).raw)
        except Exception as e:
            print("Error: Failed to process the image. Please try again.")
            exit(0)
        processor = AutoImageProcessor.from_pretrained(self.model_name)
        inputs = processor(images=image, return_tensors="pt")

        return inputs
    

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
    IMAGE_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    model = DINOV2('facebook/dinov2-base', pretrained=True)
    inputs = model.process_image(image_url=IMAGE_URL)
    traced_result = model.trace_model(inputs)
    if traced_result:
        print(f"successfully used pretrained {model.model_name}.\ntraced_result: {traced_result}")

    


