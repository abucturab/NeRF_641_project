# NeRF_641_project

## Setup the Dataset

1. Download the Cars dataset from the location [ShapeNet Cars_train](https://drive.google.com/file/d/1bThUNtIHx4xEQyffVBSf82ABDDh2HlFn/view?usp=share_link)

2. Unzip the dataset and copy the directory which contains the desired object to the Repository

3. run the data loading script 
```python dataloader.py /path/to/object/dataset```

## Training Setup

4. setup the training parameters in `NeRF/config.py`

5. run the training
```python train.py```

## Inferencing 

6. run the inference. This will create some images and a video
```python inferencer.py```

