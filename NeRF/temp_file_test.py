import os
current_path = os.path.abspath(__file__)
base_path = os.path.dirname(current_path) #..../Base_NeRF/
base_path = os.path.dirname(base_path)
IMAGE_PATH = os.path.join(base_path,"output_images") # Base_NeRF/output_images
TRAIN_JSON = os.path.join(base_path,"dataset/transforms_train.json") # Base_NeRF/dataset/transforms_train.json
VAL_JSON = os.path.join(base_path,"dataset/transforms_val.json") # Base_NeRF/dataset/transforms_val.json
TEST_JSON = os.path.join(base_path,"dataset/transforms_test.json") # Base_NeRF/dataset/transforms_test.json
DATASET_PATH = os.path.join(base_path,"dataset") # Base_NeRF/dataset/
COARSE_PATH=os.path.join(base_path, "coarse_model_store") # Base_NeRF/coarse_model_store
FINE_PATH=os.path.join(base_path, "fine_model_store") # Base_NeRF/fine_model_store
VIDEO_PATH=os.path.join(base_path, "video") # Base_NeRF/video

print(f"base_path : {base_path}")
print(f"current_path : {current_path}")
print(f"IMAGE_PATH: {IMAGE_PATH}")
print(f"TRAIN_JSON: {TRAIN_JSON}")
print(f"VAL_JSON: {VAL_JSON}")
print(f"TEST_JSON: {TEST_JSON}")
print(f"DATASET_PATH: {DATASET_PATH}")
print(f"COARSE_PATH: {COARSE_PATH}")
print(f"FINE_PATH: {FINE_PATH}")
print(f"VIDEO_PATH: {VIDEO_PATH}")