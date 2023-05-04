import json
import os
import re
from tqdm import tqdm
import random
import argparse
import sys


def getTransformDict(file_list, type="train", dir_path=None):
    my_dict = dict()
    frames_value = []
    for filename in tqdm(file_list):
        pose_path = os.path.join(dir_path, f"pose/{filename}.txt")
        image_path = os.path.join(dir_path, f"rgb/{filename}.png")
        
        if os.path.isfile(pose_path):
            scene_pose_matrix = []
            with open(pose_path, 'r') as f:
                line = f.readline().strip()
                values = [float(x) for x in line.split()]
                if len(values) != 16:
                    print(f"Invalid number of values in {filename}. Skipping.")
                    continue
                transform_matrix_entry = [values[i:i + 4] for i in range(0, 16, 4)]
            
                rotation = 0.012566370614359171
                frame_dict = dict()
                frame_dict["file_path"] = f"./{type}/{filename}"
                frame_dict["rotation"] = rotation
                frame_dict["transform_matrix"] = transform_matrix_entry
                frames_value.append(frame_dict)
        
    my_dict["camera_angle_x"] = 0.6194058656692505
    my_dict["frames"] = frames_value

    return my_dict


parser = argparse.ArgumentParser(description="Script to convert the dataset present in the ShapeNet into format compatible with this implementation of NeRF")
parser.add_argument("-p", "--path", help="Path to the directory containing image data from shapenet or in shapenet format", required=True)
args = parser.parse_args()

#print(f"Argument: {args.path}")
path_string = args.path
#STEP 1:
""" Create 2 list of images Train: 200, Val: 50"""
ShapeNet_dirname = os.path.basename(args.path)
all_files = os.listdir(os.path.join(args.path, "rgb"))
#all_files = os.listdir(os.path.join(os.getcwd(), "c951be855f20a1bfb2a29f45da811562/rgb"))
random.shuffle(all_files)



filename_list = []
for i in all_files:
    name = os.path.splitext(i)[0]
    filename_list.append(name)

train_files = filename_list[0:200]
val_files = filename_list[200:]

if os.path.exists("./dataset/train") and os.path.isdir("./dataset/train"):
    print("Directory dataset/train is already present")
    sys.exit(1)
else:
    os.makedirs("./dataset/train")

if os.path.exists("./dataset/val") and os.path.isdir("./dataset/val"):
    print("Directory dataset/val is already present")
    sys.exit(1)
else:
    os.makedirs("./dataset/val")

if os.path.exists("./dataset/test") and os.path.isdir("./dataset/test"):
    print("Directory dataset/test is already present")
    sys.exit(1)
else:
    os.makedirs("./dataset/test")

if os.path.exists("./dataset/transforms_train.json") and os.path.isfile("./dataset/transforms_train.json"):
    print("File ./dataset/transforms_train.json already present !!")
    sys.exit(1)
if os.path.exists("./dataset/transforms_val.json") and os.path.isfile("./dataset/transforms_val.json"):
    print("File ./dataset/transforms_val.json already present !!")
    sys.exit(1)
if os.path.exists("./dataset/transforms_test.json") and os.path.isfile("./dataset/transforms_test.json"):
    print("File ./dataset/transforms_test.json already present !!")
    sys.exit(1)

train_transform_dict = getTransformDict(train_files, type="train", dir_path=args.path)
val_transform_dict = getTransformDict(val_files, type="val", dir_path=args.path)

for train_pic in train_files:
    source_path = os.path.join(os.getcwd(), f"{path_string}/rgb/{train_pic}.png")
    target_path = os.path.join(os.getcwd(), "dataset/train")
    os.system(f"cp {source_path} {target_path}")

for val_pic in val_files:
    source_path = os.path.join(os.getcwd(), f"{path_string}/rgb/{val_pic}.png")
    target_path = os.path.join(os.getcwd(), "dataset/val")
    os.system(f"cp {source_path} {target_path}")
    target_path = os.path.join(os.getcwd(), "dataset/test")
    os.system(f"cp {source_path} {target_path}")

with open("./dataset/transforms_train.json", "w") as f:
    json.dump(train_transform_dict, f)

with open("./dataset/transforms_val.json", "w") as f:
    json.dump(val_transform_dict, f)

with open("./dataset/transforms_test.json", "w") as f:
    json.dump(val_transform_dict, f)