# import os
# import shutil

# # 기존 데이터 경로 및 재구성할 루트 경로 설정
# source_dir = "../../data/imagenet-mini"  # 기존 데이터 경로
# root_dir = "../../data/splitted_data"  # 새로운 루트 경로

# # train, val 데이터를 재구성
# for split in ['train', 'val']:
#     split_dir = os.path.join(source_dir, split)
#     for class_name in os.listdir(split_dir):  # 클래스 디렉토리 순회
#         class_dir = os.path.join(split_dir, class_name)
#         target_class_dir = os.path.join(root_dir, split, class_name)  # 새로운 디렉토리 경로
#         os.makedirs(target_class_dir, exist_ok=True)  # 디렉토리 생성
#         for img_file in os.listdir(class_dir):
#             # 이미지 파일 복사
#             shutil.copy(os.path.join(class_dir, img_file), target_class_dir)

# # 테스트 데이터 처리
# test_source_dir = os.path.join(source_dir, "test")
# test_target_dir = os.path.join(root_dir, "test")
# os.makedirs(test_target_dir, exist_ok=True)

# if os.path.exists(test_source_dir):  # 테스트 디렉토리가 있는 경우에만 복사
#     for img_file in os.listdir(test_source_dir):
#         shutil.copy(os.path.join(test_source_dir, img_file), test_target_dir)

# from dinov2.data.datasets import ImageNet

# # 루트 디렉토리와 메타데이터 디렉토리 설정
# root_dir = "./data/dataset/"
# extra_dir = root_dir  # 메타데이터도 같은 루트 디렉토리에 저장

# # train, val, test 메타데이터 생성
# for split in ImageNet.Split:
#     dataset = ImageNet(split=split, root=root_dir, extra=extra_dir)
#     dataset.dump_extra()

# # # train 디렉토리 기준으로 labels.txt 생성
# # train_dir = os.path.join(root_dir, "train")
# # labels_path = os.path.join(root_dir, "labels.txt")

# # with open(labels_path, "w") as f:
# #     for idx, class_name in enumerate(sorted(os.listdir(train_dir))):
# #         f.write(f"{idx}\t{class_name}\n")

import os
import shutil

# Define paths
source_dir = "../../data/imagenet-mini"  # Source directory
root_dir = "../../data"  # Target root directory
train_dir = os.path.join(root_dir, "train")
val_dir = os.path.join(root_dir, "val")
test_dir = os.path.join(root_dir, "test")
labels_file = os.path.join(root_dir, "labels.txt")

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Step 1: Check if source directories exist
source_train_dir = os.path.join(source_dir, "train")
source_val_dir = os.path.join(source_dir, "val")
# source_test_dir = os.path.join(source_dir, "test")

if not os.path.exists(source_train_dir):
    raise FileNotFoundError(f"Source train directory not found: {source_train_dir}")
if not os.path.exists(source_val_dir):
    raise FileNotFoundError(f"Source validation directory not found: {source_val_dir}")
# if not os.path.exists(source_test_dir):
#     raise FileNotFoundError(f"Source test directory not found: {source_test_dir}")

# Step 2: Get class names from the source train directory and write them to labels.txt
class_names = sorted([d for d in os.listdir(source_train_dir) if os.path.isdir(os.path.join(source_train_dir, d))])

with open(labels_file, "w") as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

# Step 3: Reorganize train and validation images into the new structure
for split in ["train", "val"]:
    source_split_dir = os.path.join(source_dir, split)
    target_split_dir = os.path.join(root_dir, split)

    for class_name in class_names:
        source_class_dir = os.path.join(source_split_dir, class_name)
        target_class_dir = os.path.join(target_split_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        if os.path.exists(source_class_dir):
            for img_name in os.listdir(source_class_dir):
                old_path = os.path.join(source_class_dir, img_name)
                new_path = os.path.join(target_class_dir, img_name)
                if os.path.isfile(old_path):
                    shutil.copy(old_path, new_path)

# Step 4: Rename and move test images
if os.path.exists(source_test_dir):
    for idx, img_name in enumerate(sorted(os.listdir(source_test_dir))):
        old_path = os.path.join(source_test_dir, img_name)
        new_name = f"ILSVRC2012_test_{idx + 1:08d}.JPEG"
        new_path = os.path.join(test_dir, new_name)
        if os.path.isfile(old_path):
            shutil.copy(old_path, new_path)

print("Dataset reorganization complete. Labels saved to labels.txt.")
