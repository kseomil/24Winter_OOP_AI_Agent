import os
import json
import sys


script_path = os.path.dirname(os.path.realpath(__file__))
IMAGE_INDEX_PATH = os.path.join(script_path, '../data/image_index.json')


def generate_image_index(image_path):
    image_dir = os.fsencode(image_path)
    image_files = os.listdir(image_dir)
    index_result = {}
    
    for i in range(len(image_files)):
        filename = os.fsdecode(image_files[i])
        if filename.endswith(".jpg") or filename.endswith(".JPEG"):
            index_result[i] = os.path.realpath(filename)

    with open(IMAGE_INDEX_PATH, 'w') as json_file:
        json.dump(index_result, json_file, indent=4)  # indent=4는 보기 좋게 들여쓰기를 추가하는 옵션입니다.


def get_image_path_by_index(index):
    with open(IMAGE_INDEX_PATH, 'r') as json_file:
        index_data = json.load(json_file)
    
    return index_data[index]
        

if __name__=="__main__":
    IMAGE_PATH = os.path.join(script_path, "../data/flickr30k/Images")
    generate_image_index(IMAGE_PATH)
    print(get_image_path_by_index('300'))