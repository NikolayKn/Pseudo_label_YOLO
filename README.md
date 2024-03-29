# YOLO training using Pseudo labeled data

The repository contains code for training the YOLO model on various markup (original data, pseudo labelled and variations).


## Datasets

I use PASCAL VOC 2007 Dataset. The PASCAL VOC (Visual Object Classes) dataset is a well-known object detection, segmentation, and classification dataset. It is designed to encourage research on a wide variety of object categories and is commonly used for benchmarking computer vision models. It is an essential dataset for researchers and developers working on object detection, segmentation, and classification tasks.

## Instructions

**Installation:**

1.Clone the repository from GitHub.

```bash
git clone git@github.com:NikolayKn/Pseudo_label_YOLO.git
```

2. Run the ```setup.sh``` script to download the Grounding DINO with model weights and install all requrements.

```bash
sh setup.sh
```

**Downloading VOC2007 dataset:**

1.Activate the environment.

```bash
source .venv/bin/activate
```

2. Run the ```download_VOC.py``` script to download the VOC2007 Dataset.

```bash
python download_VOC.py
```


## Credits
- [Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Ultralitics](https://github.com/ultralytics)  
