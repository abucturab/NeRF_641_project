import os

current_path = os.path.abspath(__file__) #..../Base_NeRF/NeRF/config.py
base_path = os.path.dirname(current_path) #..../Base_NeRF/NeRF
base_path = os.path.dirname(base_path) #..../Base_NeRF

"""Size of dataset in bytes:
    data: datatype_size * Image_dim * Image_dim * N_C * 3 * number_of_images 
    Labels: datatype_size * Image_dim * Image_dim * 3 * number_of_images"""

# if image is 512*512: float32 * 512*512 * 3 * 64 * 100 = 20GB per training dataset !!!

IMAGE_WIDTH=64 #512 Reducing to 128 as 512 is too large for Google Colab
IMAGE_HEIGHT=64 #512
IMAGE_PATH = os.path.join(base_path,"output_images") # Base_NeRF/output_images
TRAIN_JSON = os.path.join(base_path,"dataset/transforms_train.json") # Base_NeRF/dataset/transforms_train.json
VAL_JSON = os.path.join(base_path,"dataset/transforms_val.json") # Base_NeRF/dataset/transforms_val.json
TEST_JSON = os.path.join(base_path,"dataset/transforms_test.json") # Base_NeRF/dataset/transforms_test.json
DATASET_PATH = os.path.join(base_path,"dataset") # Base_NeRF/dataset/
NEAR=1.0
FAR=100.0
N_C=32
BATCH_SIZE=1
DENSE_UNITS=256
L_XYZ=10
L_DIR=4
SKIP_LAYER=5
N_F=64
EPOCHS=100
COARSE_PATH=os.path.join(base_path, "coarse_model_store") # Base_NeRF/coarse_model_store
FINE_PATH=os.path.join(base_path, "fine_model_store") # Base_NeRF/fine_model_store
STEPS_PER_EPOCH=100 #TODO review
VALIDATION_STEPS=50 #TODO review
SAMPLE_THETA_POINTS=5
FPS=30
VIDEO_PATH=os.path.join(base_path, "video") # Base_NeRF/video
QUALITY=50
MACRO_BLOCK_SIZE=16
AUTO=1